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
- [x] 1.1.1 `DiagTest`: a `Linear` whose format has no batched route on the
      active backend makes `batchReady` emit one `batch-refused` record
      naming layer, projection, format and predicate; a second layer with
      the same (projection, format, reason) does not emit again.
      WRITTEN 2026-09-06; BLOCKED on a cajeta compiler regression: cajeta
      67690686 SIGSEGVs in LLVM RAGreedy (`SplitEditor::deleteRematVictims`,
      fault +0x8) codegen-ing the test module (`--emit=exe --profile=test
      --xpu-backend=cpu`), with or without this test; the Sep 5 compiler
      (cajeta-llama-u34, 0.26.0 46b1337d) builds it. Repro:
      cajeta-llm `tmp/u1/build.log`; per-class `--emit=ir` modules all
      pass `llc -O2 -mcpu=znver5`, so the crashing function is in a module
      that snapshot never sees — `CAJETA_DUMP_CODEGEN_BC` added to name it.
      NAMED + BISECTED 2026-09-06: `opt -passes=verify` on the dumped
      WmmaKernel module: `use of undefined value '%wi.tx'` in the launch
      wrapper's inlined copy of `q4kWmmaDeqEpiKernel`'s block function —
      the work-item latch adds (`wi.next`, `wi.y.next`) still reference the
      block function's original PHIs (a detached/foreign value the inliner's
      map never saw). Reverting cajeta 66041f35 (CpuBarrierFission: latch as
      scaffold after the last barrier — the fix that newly ACCEPTS these WMMA
      kernels) makes the module compile. Fix lands in cajeta.
      FIXED 2026-09-07, cajeta 057f4fe9: the cause was not the inliner but
      the fission walk regioning the `if (t0 < rows && i0 < outDim)` join
      block twice (pre-loop and post-loop region); a barrier loop under
      divergent control flow is now DECLINED by name (the host-stub
      fallback that ran before 66041f35). Suite on that compiler, cpu:
      359 passed / 0 failed / 1 skipped, both DiagTest tests green.
- [x] 1.1.2 `DiagTest`: the `prefill-mode per-row` record carries the
      refusal count in `v1`.

### 1.2 Coding
- [x] 1.2.1 `CausalLM.batchReady` collects the first failing predicate per
      projection through a `Linear.batchRefusal()` string (`"deqPrefill
      off"`, `"no batch kernel for <ty>"`, `"rows % 128"`, `"coop cols"`,
      `"packedTy < 0"`) and emits `batch-refused` once per distinct triple.
- [x] 1.2.2 `DiagRecord` documents the two records; `LoggingDiagCallback`
      prints them.

### 1.3 Acceptance
- [x] 1.3.1 `schedthroughput <Mixtral> prompt=128 gen=1 trace` names the
      refusing tensor(s) — the measured answer to spec §4.1's first half.
      MEASURED 2026-09-06 (bench built with the Unit 1 engine): one record,
      `batch-refused attn_k q8_0: only route is the int8 Mw8 GEMM
      (prefillWeights=int8); prefillWeights=packed` (layer 0, rows 128) —
      Mixtral-8x7B Q4_K_M carries Q8_0 attention keys, so it is the same
      missing route as the 8B Q8_0; Unit 2 closes both.

## Unit 2 — Coop GEMM route on HIP for the formats without a batch kernel (spec §3)

### 2.1 TDD
- [x] 2.1.1 Spike first (no half-measures): force the coop route on HIP for
      Q8_0 and measure `schedthroughput` prefill at 512 on the 8B Q8_0
      against the 13.2 tok/s before row. The unit proceeds only if the
      spike is batched and faster; if the coop kernels misbehave on HIP the
      finding is recorded here and Unit 2 re-plans around the int8 Mw8
      route instead.
      SPIKE 2026-09-06 (`schedthroughput <8B Q8_0> prompt=512 gen=32 trace
      coophip`, bench built with cajeta 67690686): the route ENGAGES —
      `prefill-mode batched 128`, `batch-route coop ty=8 4096 4096` — and the
      GPU faults: `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on the ROCm
      queue, after which every number the process prints is garbage (first
      token 0). The coop kernels have never run on HIP before; the Vulkan
      driver would have hidden an out-of-range read (robust buffer access),
      HIP does not. Next: `CoopQuantGemmTest` on gfx1151 (the kernels' own
      parity tests) to split kernel indexing from the AMD `Tile.load` lowering.
      RESULT: the whole selftest suite built with the Sep 5 compiler runs
      on gfx1151 at 359 passed / 0 failed / 1 skipped, every
      `CoopQuantGemmTest` (Q8_0, Q2_K, Q3_K, Q5_K, Q4_0, Q5_0, and the
      non-256 widths) matching the host — the coop kernels and the AMD tile
      lowering are sound at the test shapes. The fault is in the ENGINE's
      HIP plumbing of the route (repack `wordView`, f16 staging, pad rows,
      or a missing sync) or a shape the tests never reach (128x4096x4096).
      SERIALIZED (`AMD_SERIALIZE_KERNEL=3`): NO fault, first token 77 = the
      per-row baseline's, prefill 512 in 3169 ms = 161.6 tok/s (12x the
      per-row 13.2) even with a device sync after every one of ~3300
      kernels — the coop route on HIP is CORRECT and fast; the fault is a
      RACE (a queued kernel against a buffer's lifetime or an unfinished
      transfer), deterministic unserialized (2/2), gone under
      `AMD_LOG_LEVEL=4` alone (dispatch slowed enough).
      BRACKETS (bench arms, 2026-09-07 00:xx): `pfsync` (sync at phase marks)
      → still faults; `coopsync1` (sync right after `ensureBtXh`, before the
      GEMM) → still faults; `coopsync2` (sync right after the GEMM) → the
      process printed nothing (aborted). So the fault is raised BY the coop
      GEMM dispatch (or its repack) itself when it is not preceded by a
      device-wide sync — its inputs at launch time are the suspects: the
      `packedDev` the repack reads (weight prefetch stream? handle not yet
      assigned?), `coopW` = `coopDev.wordView()`, `btXh`. Under serialization
      the same launch computes the right tokens.

      NIGHT 2 (2026-09-07, after the compiler fix landed as cajeta 057f4fe9)
      — CORRECTED: the "race" is HOST HEAP CORRUPTION on the coop route.
      Five runs of `tmp/llmbench-spike/schedthroughput <8B Q8_0> prompt=512
      gen=2 trace coophip` (same args) gave three different failures: (a)
      four runs exit 0 with every later launch saying `no registered kernel
      'gluF32' / 'q80F16CoopX1Kernel'` and first token 0 — the kernels ARE
      in the binary (`strings` shows both `.kd` descriptors; a first
      "tree-shake pruned them" reading was wrong), so the registry lookups
      themselves went bad; (b) two of those runs also page-faulted on the
      GPU (dmesg 00:24:08 / 00:24:14, client TCP, a READ of
      0x576a86f02000 — a host-heap-shaped address, i.e. a kernel handed a
      corrupted pointer); (c) the fifth run: host SIGSEGV, exit 139, fault
      addr 0x80000, symbolized against the binary:
      `__cajeta_string_drop_claimed` ← `Diag.emit` +0x11c ← `Linear.sayRoute`
      ← `matmulBatchKeep` (coop branch) — an owned string dropped inside
      `Diag.emit`, right after `prefill-mode batched` and BEFORE any coop
      launch. Heap-layout dependence is why `AMD_SERIALIZE_KERNEL=3` and
      `AMD_LOG_LEVEL=4` "fixed" it last night. Leading suspect: `sayRoute`'s
      coop call is the one that passes a HEAP TEMPORARY name (`"coop ty=" +
      this.packedTy`; dotAccum at Linear:2166 is the other), and `Diag.emit`
      does `r.name #= name` into its static scratch record and drops the
      previous value on the next emit — measure whether the temp's title is
      both tendered to the callee AND dropped by the caller (double drop),
      or a dangling borrow is later dropped as owned. `DiagTest` passes the
      same shape on CPU, so it is layout-dependent; write the probe as a
      Diag test that emits two heap-temp names back to back under a
      callback. GPU is HEALTHY (AttentionTest device tests pass, no new
      fault during the crash run). Still true: the coop route needs
      `outDim % 128 == 0` and `cols % 64|256 == 0`, so the [out=8,in=256]
      fixtures never engage it. The coopsync1/2 bench arms in cc05c21 set
      flags with NO use site in Linear (dead) — ignore their earlier
      "brackets".
      MEASURED (2026-09-07, host, no GPU): (1) `Diag.emit` is the VICTIM —
      `DiagTest.emitWithHeapTemporaryNamesLeavesTheHeapIntact` relays 256
      heap-temporary names through a plain param with same-size-class bait
      allocated between emits: every bait intact, live count balanced
      (360/0/1 both profiles, commit 457d4cf). Mechanism read in the IR +
      runtime: the caller passes the temp with transfer word 0 and drops it
      right after the call; the callee's `#=` on a String field routes a
      lend through `__cajeta_string_resolve`, which returns a FRESH wrapper
      the field owns (owned copy for len <= 256, shared stake above), then
      sets the own-bit — safe. (2) `#=` on a CLASS field from a lent param
      (`probe.Q.keep` IR): reads the arriving title flag, displaces the old
      value only if it was owned, stores, and sets own-bit = title flag —
      i.e. records a BORROW for a lend, exactly CLAUDE.md §2.3. A static
      `#=` from a param (`probe.Q.park`) just stores the pointer. The
      caller frees the object at scope end, so `Linear.btXhSrc #= src`
      (stageBatchFromDevice) DANGLES if `src` dies before the coop
      route's `ensureBtXh` reads it; the callers pass `this.pfXnDev`,
      `dkv.prefillAttendResident(...)` (= the DeviceKv FIELD `aoDev`, long-
      lived) and `d.mlp.gateProj.yBatch` (a Linear field that
      `ensureBatchOut` can REALLOCATE — check that growth path). (3) The
      bench builds cleanly THROUGH run-tests.sh's enumeration
      (`tmp/u4/build-bench.sh`: its prologue + `--emit=cja` + `--emit=exe
      --tree-shake=off --xpu-backend=amdgpu`), which is the order-dependence
      of the ownership check demonstrated by construction; note the copied
      prologue's `trap rm -rf $out EXIT` — delete it or the exe vanishes.
      (4) The 2026-09-06 leg logs and the working spike runs never printed
      `no registered kernel`; it was new to tonight's four runs.
      NAMED (2026-09-07, fresh bench = run-tests enumeration + tree-shake
      off, cajeta 057f4fe9): `no registered kernel` is NOT corruption — the
      runtime resolves kernels LAZILY (`hipModuleGetFunction` at first
      launch, cajeta_xpu_launch.c) and ignores HIP return codes; after a GPU
      fault every HIP call returns `hipErrorIllegalAddress`, so every
      not-yet-resolved kernel prints that line and the process exits 0 with
      garbage. With `AMD_LOG_LEVEL=4 AMD_SERIALIZE_KERNEL=3` (`tmp/u4/coop-
      log4.log`) the dispatch order before the first error is: blockRepack2
      → f32ToF16 (grid 2048 = 128x4096) → q80F16CoopX3Kernel (grid 32 =
      the 4096x4096 **o_proj** GEMM, first use) → addF32 launch returns
      hipErrorIllegalAddress. The q/k/v GEMMs (same kernel, same shape) ran
      fine just before. The kernel log for that pid: READ faults (client
      TCP) walking HOST-HEAP pages 0x5fe7c2551000..0x5fe7c2567000 — the
      region holding the process's HIP objects (its stream is at
      0x5fe7c2574740). So the o_proj coop launch marshals a HANDLE that is
      a host pointer: a KernelBuffer wrapper freed and its memory reused
      by libamdhip, read back as `deviceHandle` at launch — layout- and
      timing-dependent by nature, and it faults SERIALIZED too now. Diag
      trace `coop-args` (bench arm `coopargs`) prints the four handles +
      lengths per coop launch to name the argument.
      ROOT CAUSE (2026-09-07, measured in IR, `cajeta/tmp/probe-emit/src/probe/T.cajeta`):
      a local bound from a TERNARY of a borrowed field and a literal —
      `String nm = r.name != null ? r.name : "-";` — gets an ARMED drop entry
      (`__cajeta_drop_push … __cajeta_string_drop`) and frees the field's
      wrapper at scope end; `String nm = r.name;` gets none. cajeta's
      LocalVariableDeclaration classifies an initializer as a borrow only for
      the shapes it recognises (literal, identifier, field read, array
      element, plain-return call); a BooleanSwitchExpression matches none and
      defaults to OWNED. The bench's `StdoutTrace.onRecord` (trace arm) and
      the CLI's `LoggingDiagCallback.cajeta:102` both have that exact line,
      so every diag record under trace/debug logging double-frees the scratch
      record's name wrapper; the block is reused (a KernelBuffer wrapper,
      an arena tensor carrying 524288 = 128x4096 — the deterministic fault
      addr 0x80000), and the next emit frees the impostor. THAT is the whole
      "GPU race": a freed KernelBuffer wrapper's handle read back as a host
      pointer. Same class as [[kernel-ternary-mislowers]], host side. Fix in
      the COMPILER (ternary arms classified; mixed arms carry a runtime title
      flag like flagged calls); the two cajeta-llm lines are legal as written.
      FIXED IN CAJETA (2026-09-07, `fix(ownership): a local bound from a
      ternary owns exactly what the taken arm produced`, on main; tests
      TernaryOwnershipTests 4/4, 48 ownership suites 288/0/2). VERIFIED HERE:
      bench rebuilt with that compiler (`tmp/u4/build-bench.sh`, tree-shake
      off), `<8B Q8_0> prompt=512 gen=8 trace coophip`: first token 77 both
      SERIALIZED and UNSERIALIZED, no GPU fault (kernel log clean), no `no
      registered kernel`, no SIGSEGV (`tmp/u4/fix-ser3.log`, `fix-unser.log`).
      Unserialized: prefill 512 in 2013 ms = 254 tok/s, decode 27.6 tok/s —
      INDICATIVE ONLY (the cpu suite ran beside it; box not quiet); the
      per-row row of record is 13.2 tok/s. The coop route on HIP is correct;
      2.2.x (structural default) and 2.1.2–2.1.5 can proceed; 2.3.1 is the
      announced acceptance leg. The `coopargs` arm + `coop-args` Diag record
      (Linear/SchedThroughput) stay as a launch-handle instrument.
      LATER THE SAME NIGHT: a direct `cajeta --emit=cja` of the library
      (same fixed compiler, same sources, same classpath as run-tests.sh)
      FAILS with `CAJETA_ERROR_OWNED_RESULT_NEEDS_TRANSFER` at
      SafetensorsFile.cajeta:199 (`t = Tensor.zeros<float32>`), :214/:221/
      :228 (`t = this.loadF16/Bf16/F32(name)` in the *Device loaders), and
      after those are spelled `#=` a further `this.x = Tensor.zeros(...)`
      site — while run-tests.sh builds the identical library CLEAN (its
      log, line 10 → 359 passed). The ownership checker is ORDER-DEPENDENT
      (a cajeta defect: false negatives in the default order); the sites
      are genuine Producer results bound with `=`. Not fixed tonight — a
      partial one-file migration was reverted; needs a focused sweep with
      the compiler as oracle. Consequence for the instrument: build the
      bench through run-tests.sh's enumeration (or fix the sweep first),
      with `--tree-shake=off` so the coop kernels survive.
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
