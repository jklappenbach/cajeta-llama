# MXFP4 quant format — spec (Unit 32)

## 1. Definition

Add **MXFP4** (GGML tensor type `39`) to cajeta-llm's quantization
layer so MXFP4-quantized weights load and compute with the same
numerics as llama.cpp. MXFP4 is the OCP microscaling FP4 format:
32-element blocks, each a single 8-bit power-of-two scale (E8M0) plus
32 4-bit E2M1 codes. It is the on-disk weight format of gpt-oss and
other microscaling checkpoints.

### 1.1 Scope

- The `Quant` layer: type recognition, block geometry, dequantization,
  and packed matrix-vector kernels (CPU and GPU), mirroring the
  existing `Q4_K` / `Q6_K` / `Q8_0` paths in `io/Quant.cajeta`.
- GGUF/`Checkpoint` wiring so a tensor typed `39` is recognized, sized,
  and dequantized/streamed like any other quantized weight.
- A **tensor-level parity witness**: cajeta's dequant and matvec agree
  with llama.cpp's `dequantize_row_mxfp4` / `vec_dot_mxfp4_q8_0` on real
  tensors extracted from `gpt-oss-20b-MXFP4.gguf`.

### 1.2 Non-goals (explicitly a separate future unit)

- The **gpt-oss model architecture**: attention sinks, its RoPE/YaRN
  variant, MoE routing specifics, the harmony tokenizer, and
  end-to-end generation parity. Unit 32 makes the *format* correct and
  fast; bringing up the *model* is its own arc. `Checkpoint`'s hard
  `arch == "llama"` gate is left in place — an MXFP4 llama-family or
  fixture checkpoint exercises the format without gpt-oss's architecture.
- Quantizing FP32→MXFP4 (a writer). Unit 32 is read + compute only; the
  witness compares against pre-quantized llama.cpp tensors.

### 1.3 Reference

llama.cpp is the numeric oracle (`~/code/llama.cpp`):
- `ggml/src/ggml-common.h` — `block_mxfp4` (17 bytes), `QK_MXFP4=32`,
  `kvalues_mxfp4 = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}`.
- `ggml/src/ggml-impl.h` — `ggml_e8m0_to_fp32_half(e)` = `2^(e-128)`
  for `e≥2` (bits `(e-1)<<23`), denormal patterns for `e<2`.
- `ggml/src/ggml-quants.c` — `dequantize_row_mxfp4`.
- `ggml/src/ggml-cpu/quants.c` — `ggml_vec_dot_mxfp4_q8_0_generic`.

Per the fleet method note, the port must also enumerate the constraints
cajeta adds that llama.cpp does not have (§5).

## 2. Format recognition and geometry

- **2.1** When a tensor's GGML type is `39`, `Quant` reports it as a
  supported quantized type.
- **2.2** An MXFP4 block is **17 bytes** and holds **32** elements
  (1 scale byte + 16 packed-nibble bytes); `Quant` reports both.
- **2.3** A tensor whose column count is not a multiple of 32 is
  rejected with a clear error, not silently mis-sized (mirrors the
  existing K-quant width gates).

## 3. Dequantization

- **3.1** When a block is dequantized, its scale is
  `d = e8m0_to_fp32_half(e)` and element `v = kvalues_mxfp4[nibble] * d`.
- **3.2** The nibble→element mapping matches llama.cpp exactly: byte
  `qs[j]` low nibble → element `j`, high nibble → element `j + 16`
  (the two halves of the 32-wide block, **not** `2j` / `2j+1`).
- **3.3** When cajeta dequantizes a real gpt-oss-20b MXFP4 tensor, every
  value is **bit-identical** to llama.cpp's `dequantize_row_mxfp4`
  output for that tensor (the data-survival / bit-exact bar the
  gpu-numeric-fidelity arc established).

## 4. Matrix-vector kernels

- **4.1** When `y = W·x` with `W` a matrix of packed MXFP4 rows
  (row-major, cols a multiple of 32), cajeta computes it via a packed
  CPU kernel that does not first materialize a dequantized `W`, matching
  the existing `Q4_K`/`Q6_K` packed-matvec shape.
- **4.2** The kernel pairs MXFP4 weights with **Q8_0-quantized
  activations** (llama.cpp's `mxfp4 × q8_0` dot), so activation
  quantization matches the reference dot product.
- **4.3** When the same matvec runs on the GPU backend, its result
  matches the CPU kernel within the backend-parity tolerance already
  used for the other quant kernels (`BackendParityTest`).
- **4.4** A packed MXFP4 matvec against a known small fixture equals the
  dequantize-then-dense reference for the same inputs.

## 5. Constraints the cajeta port adds (measured, not assumed)

These are the port-specific hazards to check against, from prior arcs:
- **5.1** int64 `*` traps on signed overflow — scale/index arithmetic
  must not multiply into the trap; reduce-then-scale where needed
  (`rdna-quarter-rate-int-mul`, `int-overflow-traps`).
- **5.2** A `%256`/width cliff bit an earlier port on single-witness
  shapes — the width gate (§2.3) is 32, but kernels must be exercised
  on cols that are multiples of 32 but **not** of 256 as well.
- **5.3** Per-byte vs per-position reads changed nothing in one kernel
  and 22% in another — the GPU kernel's lane/byte mapping is measured,
  not inferred (`gpu-matvec-lane-mapping`, `per-byte-access-position`).
- **5.4** The CPU oracle can pass a device UAF — §4.3's GPU parity must
  run on a rebuilt compiler against the amdgpu suite, not be inferred
  from the CPU result (`cpu-oracle-passed-a-uaf`).

## 6. Witness

- **6.1** When the parity harness runs, it reads real MXFP4 tensors from
  `~/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf`
  and asserts §3.3 (dequant) and §4 (matvec) parity against a captured
  llama.cpp reference for the same tensors.
- **6.2** The reference values are produced from llama.cpp's own MXFP4
  code path (extracted once, checked in as a small fixture), so the
  witness does not depend on a live llama.cpp build at test time.
- **6.3** The witness fixture is small and shape-varied (§5.2) — at
  least one tensor whose width is a multiple of 32 but not 256.
