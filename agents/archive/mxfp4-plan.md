# MXFP4 quant format — plan (Unit 32)

**Work:** add MXFP4 (GGML type 39) read + compute support to
cajeta-llm's `Quant` layer, witnessed by tensor-level parity against
llama.cpp on real gpt-oss-20b tensors. Satisfies spec
[`mxfp4-spec`](../specs/mxfp4-spec.md). Full gpt-oss model bring-up is
out of scope (spec §1.2).

**Systems:** `src/main/cajeta/dev/cajeta/llm/io/Quant.cajeta` (format
constants, dequant, packed matvec — CPU + GPU), `io/GgufFile.cajeta` /
`io/Checkpoint.cajeta` (type wiring), the `@Kernel` device path used by
the existing Q4_K/Q6_K GPU matvecs, and the `selftest` suites
(`QuantKernelTest`, `PackedLinearTest`, `BackendParityTest`).

**Deliverables:** MXFP4 loads and dequantizes bit-exactly; a packed
MXFP4×Q8_0 matvec on CPU and GPU; a checked-in llama.cpp reference
fixture; a green parity witness over real gpt-oss-20b MXFP4 tensors.

**Reference oracle:** `~/code/llama.cpp` (spec §1.3). **Witness model:**
`~/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf`.

---

## Unit 1 — Format constants and geometry (spec §2)

### 1.1 TDD
- [x] 1.1.1 `Quant.isSupported(39)` is true; `blockSize(39) == 32`;
      block bytes `== 17`.
- [x] 1.1.2 A width not a multiple of 32 raises the width-gate error.

### 1.2 Coding
- [x] 1.2.1 `GG_MXFP4 = 39`; extend `isSupported`, block-size and
      block-byte functions; add the width gate.

### 1.3 Acceptance
- [x] 1.3.1 Existing Q4_K/Q6_K/Q8_0 geometry tests still pass unchanged.

## Unit 2 — CPU dequantization (spec §3)

### 2.1 TDD
- [x] 2.1.1 Hand-built block (known `e` + known nibbles) dequantizes to
      `kvalues_mxfp4[nibble] * 2^(e-128)` with the j / j+16 interleave.
- [x] 2.1.2 `e < 2` denormal scale path matches llama.cpp's patterns.
- [x] 2.1.3 A captured llama.cpp `dequantize_row_mxfp4` vector for a
      real tensor slice is reproduced **bit-exactly**.

### 2.2 Coding
- [x] 2.2.1 `mxfp4Block(raw, ro, out, oo)`; wire into the dequant
      dispatch beside `q4kBlock`/`q6kBlock`; the E8M0-half and E2M1
      table as constants.

### 2.3 Acceptance
- [x] 2.3.1 Bit-exact against the reference fixture (spec §3.3).

## Unit 3 — CPU packed matvec, fp32 scalar reference (spec §4.1, §4.4)

Mirrors cajeta's convention: a bit-exact **fp32-activation scalar**
kernel is the correctness reference (like `q4kMatVecIntoScalar`); the
**Q8_0-activation fast path** (spec §4.2, matching llama.cpp's
`mxfp4×q8_0` dot) is a performance optimization and moves to Unit 7.

### 3.1 TDD
- [x] 3.1.1 Packed MXFP4 fp32 matvec equals the dequantize-then-dense
      reference **bit-exactly** — same element order (0..31 per block,
      blocks in order), so the float accumulation matches (spec §4.4).
- [x] 3.1.2 A width that is a multiple of 32 but **not** 256 is correct
      (multi-block rows, spec §5.2).

### 3.2 Coding
- [x] 3.2.1 `mxfp4MatVecIntoScalar(packed, rows, cols, x, xOff, y, yOff)`
      mirroring `q4kMatVecIntoScalar`; accumulate elements in 0..31
      order so it is bit-exact against dequant-then-dense.

### 3.3 Acceptance
- [x] 3.3.1 Bit-exact against dequant-then-dense on ≥2 row/col shapes
      including one width ×32 but not ×256.

## Unit 4 — GPU packed matvec (spec §4.3)

### 4.1 TDD
- [x] 4.1.1 `BackendParityTest`: GPU MXFP4 matvec equals the CPU kernel
      within the established tolerance.
- [x] 4.1.2 Lane/byte mapping is measured on device, not inferred from
      CPU (spec §5.3–5.4): the amdgpu suite runs on the rebuilt compiler.

### 4.2 Coding
- [x] 4.2.1 `@Kernel` MXFP4 matvec mirroring the Q4_K/Q6_K device path;
      coalesced lane mapping (one wave per row).

### 4.3 Acceptance
- [x] 4.3.1 Green on the amdgpu backend suite (spec §5.4), not just CPU.

## Unit 5 — GGUF / Checkpoint wiring (spec §2.1)

### 5.1 TDD
- [x] 5.1.1 A fixture GGUF with an MXFP4 tensor loads: type recognized,
      byte size correct, dequant/stream path reached.

### 5.2 Coding
- [x] 5.2.1 Recognize type 39 in `GgufFile`/`Checkpoint` sizing and the
      weight load/stream path; route to the Unit 2/3 kernels.

### 5.3 Acceptance
- [x] 5.3.1 The real gpt-oss-20b-MXFP4 file's MXFP4 tensors are read
      without error (type + size), independent of arch support.

## Unit 6 — Correctness witness (spec §6)

### 6.1 TDD
- [x] 6.1.1 A parity harness reads real MXFP4 tensors from the
      gpt-oss-20b file and asserts dequant (§3.3) and matvec (§4)
      parity against the checked-in llama.cpp fixture.

### 6.2 Coding
- [x] 6.2.1 Extract-once tool/step producing the reference fixture from
      llama.cpp's MXFP4 path; check in a small, shape-varied fixture
      (spec §6.3 — at least one width ×32 but not ×256).

### 6.3 Acceptance
- [x] 6.3.1 The witness is green and self-contained (no live llama.cpp
      build needed at test time, spec §6.2). Correctness is
      deterministic (bit-exact) — a single run is definitive; the
      multi-run averaging discipline below is for PERFORMANCE only.

## Unit 7 — Performance parity vs llama.cpp (kernel-level)

Goal: cajeta's MXFP4 kernels **match llama.cpp** on throughput, the same
stance as the 30B parity bar. NOTE: full-model decode/prefill parity
needs the gpt-oss architecture running in cajeta (out of scope, spec
§1.2); Unit 32 measures at the KERNEL level and defers end-to-end perf
to the later gpt-oss bring-up unit.

### 7.1 TDD / measurement
- [x] 7.1.1 `bench/Mxfp4Bench` times the MXFP4 matvec on device via the
      cajeta profiler (DEVICE kernel time), and llama.cpp's mxfp4 MUL_MAT
      is timed on the SAME device/shape with `test-backend-ops perf`.
- [x] 7.1.2 Measurement discipline: profiler device time (not wall),
      warm-up discarded, 200 iters averaged, idle-gated, shape matched to
      llama.cpp (4096×14336), and the shader ISA read via `--xpu-emit=isa`
      (VGPR/spill/cndmask — the HIP path, RADV_DEBUG is inert). A
      decode-free probe isolated the memory floor from decode ALU.

### 7.2 Coding
- [x] 7.2.1 `bench/Mxfp4Bench` (scalar / fast3 / coop arms, warm-up +
      averaged) and the llama.cpp comparison via `test-backend-ops perf`
      on the matched shape. Production `matVecLaunch` routes MXFP4 to the
      portable `mxfp4MatVecKernelFast3` (int32 branchless decode, one item
      per row, correct on the CPU backend and every wave width).

### 7.3 Acceptance
- [x] 7.3.1 gfx1151, 4096×14336, profiler device time averaged over 200:
      llama.cpp mxfp4 MUL_MAT **58.4 µs** (534 GB/s, MALL-resident, HIP
      graph); cajeta **coopQ8 43 µs (1.36× FASTER than llama, 34× over
      scalar)** + a 2.3 µs one-shot activation quant; **coop (f32) 79 µs**,
      **fast3 (portable) 145–170 µs**; scalar floor ~1480 µs. PARITY MET
      AND EXCEEDED at the kernel level.
      HISTORY: the wave-cooperative layout first reached the memory floor
      (decode-free coop probe = 62 µs ≈ llama), leaving a ~100 µs
      **f32-domain decode ALU** gap (measured identical across int8-select
      / int8-branchless / int32-branchless decodes, occupancy-invariant →
      ALU-throughput-bound). Root cause: MXFP4's non-linear E2M1 table
      decoded arithmetically. FIX-1 (2026-09-04): added a **`Vector.lut4`
      byte-permute LUT intrinsic** to the compiler (lowers to `v_perm_b32`
      / `llvm.amdgcn.perm` on AMD, portable spill-and-gather elsewhere) —
      exactly llama.cpp's `get_int_from_table_16` technique. The kvalues
      (sign folded into indices 8..15) are the table, so `lut4` yields the
      decoded value directly: coop 163 → 77 µs, cndmask 32 → 1, instrs
      690 → 361. My earlier "no int8-dot op" finding was WRONG —
      `.dot`/`.dotSum` already exist; the missing op was the byte permute.
      FIX-2 (2026-09-04): the residual 77 vs 58 µs was the `lut4` output
      still running widen→toF32→f32 FMA (~15 µs). Closed it with the full
      llama.cpp algorithm — **`mxfp4MatVecKernelCoopQ8`**: `lut4` decodes to
      signed int8 and **`.dotSum` (v_dot4/sdot4 dp4a)** dots against
      **per-32 Q8_0-quantized activations** (`mxfp4QuantActKernel`, a 2.3 µs
      one-shot), scaling the integer sum by d_w·d_x once per block — no f32
      decode ALU. Result: **43 µs** (stable over 3 runs: 41.9/42.8/44.8),
      BELOW llama's 58.4 µs. ISA (`--xpu-emit=isa`) confirms the real
      hardware path: 8× `v_dot4_i32_iu8` (neg_lo realizes signed×signed) +
      24× `v_perm_b32`, zero `v_fma`, 44 VGPR / 0 spill. Correctness:
      coopQ8 differs from the f32 scalar by 0.5 % (activation-quant error,
      exactly llama's mxfp4×q8 behaviour), 0 rows > 2 %; regression-guarded
      by `Mxfp4Test.gpuCoopQ8MatchesTheHostQ8Reference` (an independent
      host Q8 twin, 2 % tolerance, amdgpu/wave32-gated). Coop/coopQ8 stay
      amdgpu/wave32-only (device can't report wave width to size a portable
      launch — the Unit 12 gap); the portable production path is fast3, and
      coopQ8 is the fastest kernel for backend-aware dispatch.
