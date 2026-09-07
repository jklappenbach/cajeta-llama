# Prefill routes — every supported quant batches on every backend

Status: draft 2026-09-06 (Julian: "fix the issues we know about first").
Evidence: the cajeta-llm vs llama.cpp sweep of 2026-09-06 (cajeta repo
`tmp/llmbench/`, memory `llm-vs-llamacpp-bench-2026-09-06`).

## 1. Definition

### 1.1 Problem

On the HIP backend, a prompt is prefilled **batched** (one GEMM per
projection per chunk) only when every attention projection of every layer
has a batched GEMM route for its quant format. Today that is true for
Q4_K and Q6_K alone. Q2_K, Q3_K_M, Q5_K_M and Q8_0 checkpoints, and Mixtral
8x7B Q4_K_M, fall to `prefill-mode per-row`: the prompt goes through the
decode mat-vec kernels one token at a time. Measured on gfx1151 against
llama.cpp 5306f4b (flash attention on, best of HIP/Vulkan):

| checkpoint | cajeta prefill 512 | llama.cpp | mode |
|---|---|---|---|
| Llama-3.1-8B Q4_K_M | 218 tok/s | 1320 | batched (`q4 plain`) |
| Llama-3.1-8B Q6_K | 218 | 1033 | batched |
| Llama-3.1-8B Q2_K | 17.9 | 1123 | per-row |
| Llama-3.1-8B Q3_K_M | 16.8 | 1349 | per-row |
| Llama-3.1-8B Q5_K_M | 15.0 | 1314 | per-row |
| Llama-3.1-8B Q8_0 | 13.2 | 914 | per-row |
| Mixtral-8x7B Q4_K_M | 16.9 | 542 | per-row |
| Qwen1.5-MoE-A2.7B Q4_K_M | timeout (300 s) | 2342 | — (load 31.6 s) |
| Qwen2.5-VL-72B Q4_K_L | timeout (300 s) | 96 | — |

Per-row is superlinear in prompt length (Q2_K: 17.9 tok/s at 512, 5.8 at
2048) and host-bound (GPU 20 % busy, one host thread at 100 %). Decode is
at parity on every one of these checkpoints (0.95–0.99x), so prefill is
the whole gap on most of the model set.

### 1.2 Cause (measured, not inferred)

- `Linear.isBatchRoutedFor` opens the Q2_K/Q3_K/Q5_K/Q8_0 batched route
  only under `Linear.deqPrefill`, which is `EngineOptions.prefillWeights ==
  "int8"` — off by default (the int8 copy doubles a weight's bytes). With
  the default `packed`, those formats have no batched route on HIP.
- The coop f16 GEMM (`QuantKernel.coopBatchLaunchNoSync`) covers every
  supported format and compiles for amdgpu (manifests exist, 134–137 VGPRs,
  no spill), but `isBatchRouted` opens it only under `backendIsVulkan()`.
- `batchReady` is all-or-nothing and silent: nothing names the tensor
  that refused. Mixtral's refusal has not been localized for that reason.
- Three shipped GEMM kernels spill (compiler manifest, gfx1151):
  `q4kWmmaDeqMw8Kernel` 124 B/lane at 256 VGPRs, `q2kWmmaDeqMw8Kernel`
  108 B, `q6kF16CoopN256GKernel` 68 B at 192. The first two are the int8
  route; the third is the coop route the fix below would put on HIP.

### 1.3 Scope

In: the HIP backend's batched prefill for every quant format the engine
loads; a diagnostic that names a refusal; the three spills; the MoE
checkpoints that prefill per-row or do not load in reasonable time.
Out: the Q4_K/Q6_K GEMM's own 5x gap to llama.cpp (arithmetic intensity
of `q4kWmmaKernel`; a tuning arc of its own), new architectures.

### 1.4 Constraints

- Numbers driven: every route change carries a before/after row from the
  same sweep recipe (`tmp/llmbench/leg.sh`), and ships only if prefill
  improves and decode is not worse beyond noise.
- Output parity: a new batched route must agree with the per-row path it
  replaces to the tolerance the coop route already met on Vulkan (teacher-
  forced perplexity within noise; greedy first token identical on the
  fixtures).
- No author-declared quantity: spill is read from the compiler's kernel
  manifest (`KernelManifest.of(name).spillBytes`), never asserted by hand.

## 2. Diagnostics

- **2.1** When `batchReady` refuses a prompt, a `batch-refused` diagnostic
  names the layer, the projection, the quant format and the predicate that
  failed, once per distinct (projection, format, reason).
- **2.2** When a prefill runs per-row, the `prefill-mode` record carries
  the refusal count beside the row count, so a bench log says why.

## 3. Batched routes for every format

- **3.1** When a Q2_K, Q3_K, Q5_K or Q8_0 projection is prefilled on HIP
  with the default `packed` weights at a shape the coop tile divides, it
  is batched through the coop f16 GEMM, and `prefill-mode` says `batched`.
- **3.2** When the prompt's last chunk is shorter than the tile, the
  chunk is padded to the tile rather than sent per-row, so no prompt
  length has a per-row tail.
- **3.3** When a format has a native int8 WMMA batch kernel (Q4_K, Q6_K),
  that route is kept; the coop route is the fallback for the formats
  without one, and the choice is recorded in `batch-route`.
- **3.4** When `prefillWeights=int8` is selected, the Mw8 routes behave as
  before; nothing in this spec changes that mode's semantics.

## 4. Mixture-of-experts checkpoints

- **4.1** When Mixtral-8x7B Q4_K_M is prefilled, the diagnostic of §2
  names the refusing tensor, and after the fix it prefills batched.
- **4.2** When Qwen1.5-MoE-A2.7B is loaded, load completes in a time
  proportional to its bytes (llama.cpp: 1.1 s for 8.8 GiB; cajeta today:
  31.6 s), and a 512-token prefill completes.
- **4.3** When Qwen2.5-VL-72B Q4_K_L is prefilled, the mixed Q4_K/Q5_K/
  Q6_K/Q8 tensors all take a batched route, and a 512-token prefill
  completes.

## 5. No shipped GEMM kernel spills

- **5.1** When the engine is built for gfx1151, `q4kWmmaDeqMw8Kernel`,
  `q2kWmmaDeqMw8Kernel` and `q6kF16CoopN256GKernel` record
  `spillBytes = 0` in their kernel manifests, and a selftest reads that
  from `KernelManifest.of(...)` rather than from a build log.
- **5.2** When a spill is removed, the kernel's measured duration on its
  route is not worse beyond noise, and the route's prefill tok/s row is
  recorded beside the before row.

## 6. Acceptance (whole spec)

- **6.1** The 2026-09-06 sweep recipe rerun on the same box: no
  `prefill-mode per-row` on any checkpoint in §1.1; every prefill ratio
  to llama.cpp at or above the Q4_K_M row's (0.17x); decode within noise
  of its before row; load not worse.
- **6.2** The `selftest` suite green on the CPU backend and on gfx1151.
