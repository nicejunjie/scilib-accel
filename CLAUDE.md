# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

`scilib-accel` is an `LD_PRELOAD` library that intercepts BLAS calls from CPU applications and offloads them to the GPU via cuBLAS / cuSolver, targeting NVIDIA Grace-Hopper's unified-memory architecture. All 30 Level-3 BLAS routines are intercepted. The point of the library is to make pre-existing CPU codes (PARSEC, MuST, HPL, ...) run on the GPU **without source modification** — just `LD_PRELOAD=scilib-dbi.so ./your_binary`.

Two interception mechanisms exist:
- `scilib-dbi.so` — FRIDA dynamic binary instrumentation. Works with statically *or* dynamically linked BLAS. **Recommended.**
- `scilib-dl.so` — classic `dlsym(RTLD_NEXT, ...)`. Works only with dynamically linked BLAS but has no FRIDA dependency.

Build with `make dbi`, `make dl`, or `make` (both + `test_dgemm.x`). Needs NVHPC (`pgcc`/`pgf90`/`nvcc`) and cuBLAS/cuSolver.

## How it's organised

```
scilib-accel/
├── main-dbi.c, main-dl.c     FRIDA / dlsym entry points (.init_array/.fini_array)
├── init.c, global.h          env-var parsing (SCILIB_*)
├── freplace.h                 the 72-entry table of BLAS symbols to intercept
├── nvidia.c, nvidia.h        cuBLAS/cuSolver handle + stream lifecycle
├── blas/NVIDIA/*.c           30 wrappers (one per Level-3 routine)
├── utils/
│   ├── utils.c               move_numa, which_numa, HBM-aware malloc, timers
│   ├── utils.h
│   ├── gpu_migrate.cu        experimental GPU page-touch kernel
│   └── utils-mpi.c
├── test_dgemm.f90            single-process dgemm micro-test (UNRELIABLE — see below)
├── test_ztrsm.f90
├── proxy/                    multi-rank fast benchmark (closer to real workloads)
├── quick-test/               (sibling) full MuST/LSMS test harness
└── NOTES_HBM_MALLOC.md       record of the HBM-malloc + SCILIB_MV work
```

Parent `../CLAUDE.md` has the original architecture / migration-history write-up; don't duplicate it here.

## Data-movement strategy

Three offload modes selectable via `SCILIB_OFFLOAD_MODE`:
- **S1** (`=1`) `cudaMallocAsync`/`cudaMemcpyAsync` H2D, compute, D2H. Works on any GPU.
- **S2** (`=2`) Unified access; pin matrices on HBM, no explicit migration. Grace-Hopper only.
- **S3** (`=3`, **default**) "GPU First Use": on each BLAS call, check NUMA location, migrate DRAM→HBM if needed, run cuBLAS directly on the same pointer. Best on Grace-Hopper for typical reuse-heavy workloads.

In S3, the per-BLAS-call migration goes through `move_numa()` in `utils/utils.c`.

## What changed recently (the headline)

Two layered improvements to the data-movement path, both on by default and toggleable:

### 1. HBM-aware malloc interposer (the big win)

`utils/utils.c` now exports `malloc`/`calloc`/`realloc`/`free`. Allocations ≥ a threshold are `mbind`-ed to HBM at allocation time, so by the time a BLAS call arrives the pages are already on HBM and `move_numa` becomes a no-op for them.

- Env: `SCILIB_HBM_MALLOC_MB`
  - unset (default) → **on**, 64 MB threshold
  - `0` → off (interposer still in front of glibc, but no `mbind`)
  - `N>0` → on, N MB threshold

Below the threshold, small allocs pass through unchanged so CPU-side scratch keeps DRAM locality.

### 2. `SCILIB_MV` switch inside `move_numa`

Six implementations of the page-migration step are now selectable at runtime:

| `SCILIB_MV` | implementation | notes |
|---|---|---|
| `0` (default) | `move_pages` syscall | state-of-the-art on real workloads |
| `1` | `mbind(MPOL_BIND \| MPOL_MF_MOVE)` parallel chunks | ties `0` on real workloads |
| `2` | `cudaMemPrefetchAsync` on cuBLAS stream | 3× faster on test_dgemm, **2× slower on run2.sh** |
| `3` | `cudaMemAdvise(SetPreferredLocation)` + prefetch | same as 2 |
| `4` | no-op (rely on hardware coherence) | wins on tiny tests, 2.2× slower on run2.sh |
| `5` | GPU page-touch kernel | 2.2× slower on run2.sh |
| `6` | VMA-wide `mbind` on first touch | cuts migration time but kills CPU access — 1.8× slower |

**Do not change the default unless you're A/B testing**. None of variants 1–6 has been shown to beat 0 on the real workload.

## Headline numbers (quick-test/run2.sh, MuST/LSMS, CoCrFeMnNi, 28 MPI × OMP=2)

```
config                                  app wall (s)   speedup vs CPU
Pure CPU baseline (no LD_PRELOAD)          126.7          1.00×
scilib-accel, HBM-malloc OFF                32.2          3.93×
scilib-accel default (HBM-malloc 64 MB)     26.5          4.79×
```

Physics output matches the reference `o_n0000000_CoCrFeMnNi.ref` byte-for-byte. See `NOTES_HBM_MALLOC.md` for the full investigation, profile data, and what was ruled out.

## Don't trust `test_dgemm.x` to validate `move_numa` changes

`test_dgemm.x` is a single-process micro-test (5 iterations of the same dgemm on one reused buffer). It badly mispredicts real-workload behaviour for any change to the data-movement path:

| variant | `test_dgemm` says | `run2.sh` actually |
|---|---|---|
| `SCILIB_MV=2` (prefetch) | 3× faster | 2× **slower** |
| HBM-malloc 64 MB | hard to tell | 17 % **faster** |
| `SCILIB_MV=4` (no-op) | 5× faster (wall) | 2.2× **slower** |

Reasons: `test_dgemm` doesn't expose 28-rank GPU/driver contention, doesn't expose CPU-side regressions from HBM-resident data, and amortises migration across only 5 calls.

**Always validate on `quick-test/run2.sh` (or `proxy/run_proxy.sh`) before declaring a win.**

## How to verify a change

Real workload (~30 s / run; 3 reps by default):

```bash
cd quick-test
./run2.sh                                # default config
SCILIB_HBM_MALLOC_MB=0 ./run2.sh         # interposer off
SCILIB_MV=2 ./run2.sh                    # try a move_numa variant
NRUNS=5 ./run2.sh                        # more reps
```

Outputs: `o_n0000000_CoCrFeMnNi.run<N>` (physics output, "Job total" line) and `log.run<N>` (per-rank BLAS timing).

Fast proxy (~10 s / run; same launch shape, less compute, magnifies migration effects):

```bash
cd proxy && make
./run_proxy.sh                # baseline
./run_proxy.sh 3 0 64         # HBM-malloc 64 MB
./run_proxy.sh 3 2            # try cudaMemPrefetchAsync
```

The proxy reproduces the **sign** of each variant change (prefetch → slower, HBM-malloc → faster); trust the ordering, validate magnitudes on `run2.sh`.

## Env vars

| variable | purpose | default |
|---|---|---|
| `SCILIB_HBM_MALLOC_MB` | HBM-aware malloc threshold (0 = off) | on, **64 MB** |
| `SCILIB_MV` | `move_numa` variant 0–6 | `0` (`move_pages`) |
| `SCILIB_OFFLOAD_MODE` | S1/S2/S3 strategy | `3` |
| `SCILIB_MATRIX_OFFLOAD_SIZE` | offload threshold as cbrt(m·n·k) | `500` |
| `SCILIB_OFFLOAD_FUNC` | comma-list of routines to intercept | all |
| `SCILIB_DEBUG` | 0–3 verbosity | `0` (timing summary still printed at fini) |
| `SCILIB_THPOFF` | turn off THP via prctl | `0` |
| `SCILIB_HBM_NUMA` | target NUMA node for HBM | `1` |

## Known caveats

- After a run, `nvidia-smi` shows ~1.5 GiB still "used" on HBM even with no compute process running. **This is the CUDA driver's persistent-mode context cache, not a leak.** It does not grow with repeated runs; `cudaDeviceReset` does not free it; recovery needs `sudo nvidia-smi -pm 0/1` or `--gpu-reset`. Pre-dates the HBM-malloc work — confirmed by running with `SCILIB_HBM_MALLOC_MB=0` (residual is unchanged).
- OpenMPI+UCX requires `--mca coll ^hcoll` (already wired into `quick-test/run2.sh`).
- THP on the test node is `never` — hugepage optimisations don't apply without a system-level change.

## Repo state at the end of this session

- Modified, not committed: `utils/utils.c`, `quick-test/run2.sh`, `quick-test/exe.sh`.
- New, not committed: `proxy/` (proxy benchmark), `NOTES_HBM_MALLOC.md`, this `CLAUDE.md`.
- Default behaviour change: HBM-aware malloc on at 64 MB. Set `SCILIB_HBM_MALLOC_MB=0` to revert to the previous baseline.
