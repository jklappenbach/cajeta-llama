# Prefill routes — plan (Unit 33)

Implements [`specs/prefill-routes-spec.md`](../specs/prefill-routes-spec.md).
Draft 2026-09-06; Julian: "let's fix the issues we know about first".

**Work:** give every quant format a batched prefill route on HIP, make a
batched refusal name itself, remove the three GEMM spills, and bring the
two MoE checkpoints that prefill per-row or time out onto the batched path.
Every route change is gated by the sweep recipe that found the problem.

**Systems:** `model/Linear.cajeta` (`isBatchRouted`, `isBatchRoutedFor`,
`matmulBatchKeep` route chain, `sayRoute`), `model/CausalLM.cajeta`
(`batchReady`, `prefill-mode` diag), `io/QuantKernel.cajeta`
(`coopBatchLaunchNoSync`, `coopColsOk`, `hasBatchKernel`),
`io/WmmaKernel.cajeta` (the Mw8 kernels), `model/ExpertBank.cajeta`,
`DiagRecord.cajeta`; stdlib `cajeta.xpu.KernelManifest` (spill oracle);
the cajeta repo's `tmp/llmbench/leg.sh` + `summary.sh` (the sweep recipe)
and its `rows.jsonl` of 2026-09-06 as the before rows.

**Deliverables:** `batch-refused` diagnostic; coop GEMM route on HIP for
Q2_K/Q3_K/Q5_K/Q8_0 with a padded tail chunk; Mixtral and Qwen1.5-MoE
batched; three kernels at `spillBytes = 0`; a before/after table in this
plan's acceptance and in the bench memory.

---

## Unit 1 — Name the refusal (spec §2)

### 1.1 TDD
- [ ] 1.1.1 `DiagTest`: a `Linear` whose format has no batched route on the
      active backend makes `batchReady` emit one `batch-refused` record
      naming layer, projection, format and predicate; a second layer with
      the same (projection, format, reason) does not emit again.
- [ ] 1.1.2 `DiagTest`: the `prefill-mode per-row` record carries the
      refusal count in `v1`.

### 1.2 Coding
- [ ] 1.2.1 `CausalLM.batchReady` collects the first failing predicate per
      projection through a `Linear.batchRefusal()` string (`"deqPrefill
      off"`, `"no batch kernel for <ty>"`, `"rows % 128"`, `"coop cols"`,
      `"packedTy < 0"`) and emits `batch-refused` once per distinct triple.
- [ ] 1.2.2 `DiagRecord` documents the two records; `LoggingDiagCallback`
      prints them.

### 1.3 Acceptance
- [ ] 1.3.1 `schedthroughput <Mixtral> prompt=128 gen=1 trace` names the
      refusing tensor(s) — the measured answer to spec §4.1's first half.

## Unit 2 — Coop GEMM route on HIP for the formats without a batch kernel (spec §3)

### 2.1 TDD
- [ ] 2.1.1 Spike first (no half-measures): force the coop route on HIP for
      Q8_0 and measure `schedthroughput` prefill at 512 on the 8B Q8_0
      against the 13.2 tok/s before row. The unit proceeds only if the
      spike is batched and faster; if the coop kernels misbehave on HIP the
      finding is recorded here and Unit 2 re-plans around the int8 Mw8
      route instead.
- [ ] 2.1.2 `LinearKernelRouteTest`: on gfx1151, a Q8_0 / Q2_K / Q3_K /
      Q5_K `Linear` built from the `kquant/` fixture blocks reports
      `isBatchRoutedFor(128) == true` with `prefillWeights=packed`, and the
      batched output matches the per-row (forced serial) output within the
      coop route's Vulkan tolerance, per format.
- [ ] 2.1.3 `LinearKernelRouteTest`: Q4_K and Q6_K still take their native
      int8 routes (`batch-route q4 plain` / `mmq q6`); the coop route is not
      chosen where a native kernel exists.
- [ ] 2.1.4 `ForwardTest`: a 200-token prompt (128 + 72 tail) prefills
      `batched` end to end — the tail chunk is padded, not sent per-row —
      and the logits of the last real row equal the unpadded per-row
      logits within tolerance.
- [ ] 2.1.5 `EngineTest`: `prefillWeights=int8` still selects the Mw8
      routes for the formats that have them (spec §3.4 — nothing regresses).

### 2.2 Coding
- [ ] 2.2.1 `Linear.isBatchRouted`: open the coop branch on every backend
      (not only Vulkan) for formats where `!hasBatchKernel(ty)`, behind
      `coopColsOk` and `outDim % 128`; `isBatchRoutedFor` stops requiring
      `deqPrefill` for those formats when the coop route is open.
- [ ] 2.2.2 `matmulBatchKeep`: the coop branch is reachable on HIP;
      `sayRoute` records `coop <fmt>`; the f32→f16 activation conversion and
      the coop scratch (`yBatch`, staging) are allocated on the HIP device
      path exactly as on Vulkan.
- [ ] 2.2.3 Tail padding: `CausalLM.forwardRowsBatched` (or the chunk
      planner) rounds the last chunk's rows up to the route's tile
      (`fitsMw8`/coop tile) with zero rows, and the epilogue ignores them.
- [ ] 2.2.4 `EngineOptions` doc: `packed` now batches every format; the
      `int8` note names the Mw8 route as the alternative, not the only one.

### 2.3 Acceptance
- [ ] 2.3.1 Sweep legs (`leg.sh cajeta <8B Q2_K|Q3_K_M|Q5_K_M|Q8_0>
      512x128 3` and `2048x64 3`): `prefill-mode batched`, prefill tok/s
      ≥ 0.17x the best llama.cpp row for each, decode within noise of the
      2026-09-06 rows. Rows appended to the bench memory table.
- [ ] 2.3.2 Teacher-forced perplexity (`PplProbe`) on the 8B Q8_0 within
      noise of the per-row run.

## Unit 3 — The MoE checkpoints (spec §4)

### 3.1 TDD
- [ ] 3.1.1 `MoeForwardTest` (fixture `toy-moe.gguf` with an expert format
      that had no batched route): prefill is `batched`; expert-group
      dispatch records `device` for the groups the budget admits.
- [ ] 3.1.2 A load-time test on `toy-moe.gguf`: `LlmEngine.load` wall is
      dominated by bytes read (an instrumented breakdown — pack, upload,
      warm-up — is printed under `trace`), so the Qwen1.5-MoE 31.6 s has a
      named component before it is fixed.

### 3.2 Coding
- [ ] 3.2.1 Fix whatever Unit 1's diagnostic names on Mixtral (expected:
      an attention projection format without a HIP route, closed by
      Unit 2 — verify, do not assume).
- [ ] 3.2.2 Qwen1.5-MoE: profile the load (60 experts × 24 layers of small
      slabs; shared expert; `attn_*.bias`) and remove the component that
      scales with expert count rather than bytes; then the 512 prefill.
- [ ] 3.2.3 Qwen2.5-VL-72B Q4_K_L: with Unit 2 in place, confirm every
      tensor format routes; fix the remaining refusal if the diagnostic
      names one.

### 3.3 Acceptance
- [ ] 3.3.1 Sweep legs for Mixtral, Qwen1.5-MoE (load ×3 + 512x128) and
      the 72B (512x128 ×1): no per-row, no timeout, load ratio vs
      llama.cpp recorded.

## Unit 4 — No shipped GEMM kernel spills (spec §5)

### 4.1 TDD
- [ ] 4.1.1 `QuantKernelTest`: `KernelManifest.of("q4kWmmaDeqMw8Kernel")`,
      `("q2kWmmaDeqMw8Kernel")`, `("q6kF16CoopN256GKernel")` report
      `spillBytes == 0` on gfx1151 (skips where the backend has no
      footprint); the same test lists every registered GEMM kernel and
      fails on any non-zero spill, so the check cannot silently narrow.
- [ ] 4.1.2 Each kernel's existing correctness test still passes
      bit-for-bit (they are int8/f16 tile kernels with exact references).

### 4.2 Coding
- [ ] 4.2.1 `q4kWmmaDeqMw8Kernel` (256 VGPR, 124 B): cut live registers —
      the eight persistent f32 accumulators (~64 VGPRs) are the named
      price; drain half per chunk or narrow the N tile — ISA-verified
      (`cajeta --xpu-emit=isa`, `vgpr_spill_count = 0`).
- [ ] 4.2.2 `q2kWmmaDeqMw8Kernel` (256 VGPR, 108 B, 25 KB LDS): same
      treatment; LDS is its occupancy limiter, so the register cut must
      not move work into LDS.
- [ ] 4.2.3 `q6kF16CoopN256GKernel` (192 VGPR, 68 B): the coop route
      Unit 2 puts on HIP — fix before Unit 2's acceptance leg on Q6_K-
      bearing formats, or record that Q6_K keeps its native route.

### 4.3 Acceptance
- [ ] 4.3.1 Before/after duration on each kernel's own route (`MmqProbe` /
      `DecodeProbe` shape it runs at), not worse beyond noise; prefill
      tok/s row for the routes that use them.

## Unit 5 — Whole-spec acceptance (spec §6)

### 5.1 TDD
- [ ] 5.1.1 `run-tests.sh` green on CPU and gfx1151.

### 5.2 Coding
- [ ] 5.2.1 Update `llm-vs-llamacpp-bench-2026-09-06` memory and the cajeta
      repo's bench page with the after rows; archive this plan.

### 5.3 Acceptance
- [ ] 5.3.1 The 2026-09-06 sweep recipe rerun: no `per-row` on any
      checkpoint; every prefill ratio ≥ 0.17x; decode within noise; load
      not worse. Table in this section.
