# NVHPC 26.1 "zgemm regression" on MuST — root cause study

**Date:** 2026-06-10/11
**Node:** Vista GH200 120GB (72-core Grace, 1 GPU), THP off
**Workload:** `quick-test/run2.sh` — MuST (mst2), CoCrFeMnNi, 28 MPI ranks × OMP=2, MPS on
**Question:** is `nvhpc-hpcx-cuda13/26.1` as fast as the production `nvhpc-hpcx-cuda12/25.3`?

## TL;DR

26.1 is **~19% slower** out of the box (27.6 s vs 23.2 s wall). The cause is **not
cuBLAS 13 / CUDA 13** and not NVPL BLAS kernel speed. It is **NVPL LAPACK 26.1**,
which changed its zgetrf blocking and now emits a ladder of thin rank-24 zgemm
panel updates whose `cbrt(m·n·k)` score falls below scilib-accel's offload
threshold (500), so ~354 GFLOP/rank of new GEMM work lands on 2 CPU cores per rank.

**Workaround (verified):** `SCILIB_MATRIX_OFFLOAD_SIZE=300` under 26.1 →
23.6–23.8 s, within ~2% of the 25.3 baseline.

**Recommendation:** stay on 25.3 with defaults; if 26.1 is required, set the
threshold to 300. Longer term, replace the cbrt criterion with a flops-aware one
(thin-k GEMMs are undervalued by the geometric mean of dimensions).

## Headline numbers

| config                                            | wall (s) | zgemm t1 /rank | ztrsm t1 /rank |
|---------------------------------------------------|----------|----------------|----------------|
| 25.3 full stack (baseline)                        | 23.16    | 10.7           | 4.05           |
| 26.1 full stack                                   | 27.6     | 19.1           | 3.07           |
| 25.3 stack + **NVPL 26.1 only** (overlay)         | 27.7–27.8| 19.1           | 3.08           |
| 25.3 stack + NVPL 26.1 **BLAS only**              | 23.0     | 10.7           | 3.74           |
| 25.3 stack + NVPL 26.1 LAPACK only                | fails: LAPACK 26.1 needs `ztrsm_internal_` from BLAS 26.1 |
| 26.1 NVPL + `SCILIB_MATRIX_OFFLOAD_SIZE=300` + MPS| **23.6–23.8** | 14.0      | 4.64           |

(t1 = "compute" accumulator of the intercepted symbol, which on the CPU path is
just the original NVPL call time; CPU and GPU calls fold into the same counter.)

Physics output under 26.1 matches the reference except two values differing by
1 in the 11th decimal (normal BLAS version-to-version rounding).

## What was ruled out, and how

1. **cuBLAS 13 zgemm** — standalone `test_zgemm_gpu.c` (cublasZgemm on
   HBM-resident system memory, mst2's exact offloaded shapes, lda=6750):
   identical between CUDA 12.8 and 13.1 (39–58 TFLOPS depending on shape).
2. **NVPL BLAS tiny-zgemm speed** — `test_zgemm_cpu.c` at the two dominant CPU
   shapes (25³: 4.6 µs/call, 50³ with ldc=6750: 17.4 µs/call): identical between
   NVPL 25.3 and 26.1, including bound to 2 cores (`taskset`) and 28 concurrent
   bound instances saturating the node.
3. **MPI / hpcx version, scilib build compiler** — the regression reproduces with
   the 25.3-built scilib-dbi.so, 25.3 MPI, 25.3 CUDA, swapping *only*
   `libnvpl_*` via an `LD_LIBRARY_PATH` overlay directory of symlinks.

## Root cause detail

NVPL 26.1 restructured the LAPACK→BLAS interface and its blocking:

- **`*_internal_` symbols.** `libnvpl_lapack_core.so.0` 26.1 calls BLAS through
  new exported symbols (`zgemm_internal_`, `ztrsm_internal_`, ...) provided by
  NVPL BLAS 26.1 (absent in 25.3 — the two halves cannot be mixed across
  versions). scilib-accel's `freplace.h` already hooks all 30 `_internal_`
  variants to the same `my*` wrappers, and each wrapper hardcodes the public
  findex (`fi = zgemm`), so interception still works and the time folds into the
  public symbol's stats. Interception is NOT the problem.
- **Blocking change in zgetrf (mst2's block-LU path).** Call histograms from
  `SCILIB_DEBUG=3` runs (counts are totals across 28 ranks):
  - 25.3: trailing updates `m=n∈{907..1977}, k=256`, 3,696 calls — all above
    threshold, offloaded to GPU. 30.48 M tiny CPU calls (24.3 M of 25³, 6 M of 50³).
  - 26.1: trailing updates become `m=n∈{1465..2041}, k=192` (2,016 calls, still
    offloaded) **plus a new ladder of rank-24 updates** — shapes like
    1459×1435×24 stepping down by 24, **154,042 calls ≈ 9,916 GFLOP total
    ≈ 354 GFLOP/rank**. Tiny-call population is unchanged (30.67 M).
- The k=24 ladder scores `cbrt(m·n·k)` ≈ 95–369 < 500, so scilib left those on
  the CPU (2 cores/rank) → ≈ +8.4 s/rank in zgemm, +4.5 s wall. The ztrsm
  "improvement" under 26.1 (4.05 → 3.07 s) is the same algorithm change moving
  work between routines, not a cuBLAS win.

## Mitigation

`SCILIB_MATRIX_OFFLOAD_SIZE=300` offloads the big end of the k=24 ladder
(19,488 GPU k=24 calls across ranks) → 23.6–23.8 s with MPS. Residual ~2% vs
25.3 is the cost of those calls on a 28-rank-shared GPU.

Note: an earlier threshold test that bypassed `run2.sh` ran **without MPS** and
showed no benefit (29.4 s) — MPS matters for this mitigation; always go through
`run2.sh`.

## Follow-up idea (not implemented)

Replace the offload criterion `cbrt(m·n·k) ≥ N` with a flop-count threshold
(e.g. `8·m·n·k ≥ F`, optionally with a min-dimension guard). The cbrt metric
rates 1459×1435×24 (~0.4 GFLOP) like a small matrix; any future library that
changes blocking can re-trigger this class of regression at a different size.

## Addendum 2026-06-11: further tuning on the 25.3 production stack

Two additional wins found and verified (run2.sh, MPS on, 2 reps each):

| config (all 25.3)                              | wall (s) | zgemm t1 | ztrsm t1 |
|------------------------------------------------|----------|----------|----------|
| baseline (threshold 500, cublasZgemm)          | 23.16    | 10.7     | 4.05     |
| `SCILIB_MATRIX_OFFLOAD_SIZE=250`               | 22.55–22.64 | —     | —        |
| threshold 250 + **GEMM3M**                     | **21.70–21.77** | 9.93 | 3.35  |

Total: **−1.45 s (~6.3%)** vs the previous best.

- **Threshold 250**: offloads mid-size CPU calls like 25900×25×25 (cbrt ≈ 253,
  ~0.6 s/rank on CPU). ~0.6 s wall win.
- **GEMM3M**: GPU complex GEMMs use `cublasZgemm3m`/`cublasCgemm3m` (3-mult
  complex GEMM, ~25% faster). ~0.9 s wall win; ztrsm also speeds up
  (3.4 vs 4.05 s) because the GPU is less congested. **Physics matches the
  reference byte-for-byte.** Now a **runtime env var `SCILIB_GEMM3M` (default 0,
  off)** — was a compile-time `#define`; converted so one .so does both. `=1`
  enables it; default-off `cublas[CZ]gemm` gives 22.6 s and differs from the
  GEMM3M reference by one last-ulp. Implemented as a ternary macro in
  `blas/NVIDIA/{z,c}gemm.c` branching on the `scilib_gemm3m` global parsed in
  `init.c`; call sites unchanged.
- Threshold 250 is set via env var; the library default is still 500 — change
  `init.c` if it should become the default. (Single-node mpirun inherits the
  exported environment, so no `-x` forwarding is needed for any SCILIB_* var.)

**26.1 + GEMM3M + threshold 300** (26.1 rebuild, `scilib-dbi.so.26.1-gemm3m`):
22.9–24.2 s (mean 23.6, noisy; zgemm t1 13.7 s). GEMM3M helps 26.1 too, but the
LAPACK blocking change keeps it ~1.2 s behind the 25.3 best — 25.3 remains the
recommendation.

**`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` sweep** (on 25.3 + GEMM3M + threshold 250):
10% → 24.69 s, 25% → 23.16 s, 50% → 22.29 s,
75% → 22.05 s, unset/100% → **21.7 s**. Monotonic: any SM cap hurts. Each rank
issues one synchronous cuBLAS call at a time, so kernels benefit from the whole
GPU for their short burst; MPS already provides the overlap. **Leave it unset.**

Not yet explored: rank×thread layouts using all 72 cores (28×2 = 56; MuST's
decomposition constrains rank count, e.g. `RANK=14 NT=5` would use 70 cores),
flops-aware offload criterion.

### Bottom line — best known configuration (21.7 s, was 23.16 s)

Just run `quick-test/run_best.sh` — it pins all of the below. Equivalent by hand:

```bash
module load nvhpc-hpcx-cuda12/25.3          # NOT 26.1
export SCILIB_GEMM3M=1                       # default 0 (off); 1 = cublas[CZ]gemm3m
export SCILIB_MATRIX_OFFLOAD_SIZE=250        # library default is still 500
# CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: leave UNSET (any cap is slower)
cd quick-test && ./run2.sh                   # MPS on by default
```

The hard floor is ~7.8 s/rank of 25³/50³ zgemms (~1.1 M calls/rank) — too small
to offload individually; only app-side batching or a batched-interception scheme
could attack that.

## Artifacts

- `quick-test/run_best.sh` — one-command runner for the best-known config:
  switches to nvhpc 25.3 if needed, verifies the deployed scilib-dbi.so has
  GEMM3M (warns if not), sets `SCILIB_MATRIX_OFFLOAD_SIZE=250`, unsets the MPS
  thread cap, then delegates to run2.sh (all its overrides still work).

- `quick-test/test_zgemm_cpu.c`, `quick-test/test_zgemm_gpu.c` — micro-benchmarks
  (CPU one builds with `pgcc -O2 -nvpl -mp`; GPU one with `pgcc -O2 -cuda -lcublas`).
- `quick-test/nvpl-253-only/`, `nvpl-261-only/`, `nvpl-261-blas/`,
  `nvpl-261-lapack/` — symlink overlay dirs for `LD_LIBRARY_PATH` A/B swaps.
- `quick-test/log.run*.nvhpc261`, `o_n0000000_CoCrFeMnNi.run*.nvhpc261` — full
  26.1 run logs.
- The deployed `scilib-dbi.so` is a 25.3 build with the runtime `SCILIB_GEMM3M`
  switch (GEMM3M is no longer compile-time). Old `.so.25.3` / `.so.26.1` /
  `.so.*-gemm3m` snapshots remain from the A/B work.
- `utils/utils.c` now has a `#if CUDART_VERSION >= 13000` guard: CUDA 13 changed
  `cudaMemAdvise`/`cudaMemPrefetchAsync` to take `struct cudaMemLocation`
  (affects only the experimental `SCILIB_MV=2/3` variants).
- `quick-test/run2.sh` no longer does `-x` forwarding: single-node mpirun
  inherits the exported environment, so any SCILIB_* / CUDA_MPS_* var set in the
  calling shell reaches every rank automatically.
