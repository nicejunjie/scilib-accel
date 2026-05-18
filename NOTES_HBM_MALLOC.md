# HBM-aware malloc + move_numa variant switch — notes

Self-contained record of the work done in this session. Pairs with the parent
`auto-offload/CLAUDE.md` (architecture) and `proxy/README.md` (fast bench).

## TL;DR

Two changes to `scilib-accel`:

1. **HBM-aware `malloc` interposer**, on by default at 64 MB threshold.
   Allocations ≥ threshold are `mbind`-ed to HBM (NUMA node 1) at allocation
   time, so the BLAS wrappers never have to migrate them. Below the threshold,
   `malloc` is transparent so CPU-hot small objects keep DRAM locality.

2. **`SCILIB_MV` switch** inside `move_numa`, exposing six implementations
   for A/B testing (default = `move_pages`, the previous state-of-the-art).

Result on `quick-test/run2.sh` (MuST/LSMS, CoCrFeMnNi, 28 MPI ranks × OMP=2):

| configuration                                  | app wall (s) | speedup vs CPU |
|------------------------------------------------|--------------|----------------|
| Pure CPU 28×2 (no scilib-accel)                | 126.69       | 1.00×          |
| Pure CPU 56×1 (no scilib-accel)                | ~124  (n=1)  | ~1.02×         |
| scilib-accel, HBM-malloc **off**               | 32.20        | 3.93×          |
| **scilib-accel default (HBM-malloc on, 64 MB)**| **26.46**    | **4.79×**      |

The HBM-malloc default takes the GPU-offload speedup from 3.9× → 4.8×.

Physics output matches the reference `o_n0000000_CoCrFeMnNi.ref` byte-for-byte
in all configurations tested.

## Env vars (new and modified)

| variable                | values                                              | default                          |
|-------------------------|-----------------------------------------------------|----------------------------------|
| `SCILIB_HBM_MALLOC_MB`  | `0` = off, `N>0` = bind allocs ≥ N MB to HBM        | on, threshold **64 MB**          |
| `SCILIB_MV`             | `0..6` — pick `move_numa` implementation (see below)| `0` (`move_pages`)               |
| `SCILIB_DEBUG`          | unchanged (`0..3`)                                  | unchanged                        |

`SCILIB_MV` values:

- `0` `move_pages` — state-of-the-art baseline (previous default)
- `1` `mbind`     — `mbind(MPOL_BIND | MPOL_MF_MOVE)`, parallel chunks
- `2` `cudaMemPrefetchAsync` — on `scilib_cuda_stream`
- `3` `cudaMemAdvise(SetPreferredLocation)` + prefetch
- `4` no-op (rely on Grace-Hopper coherence)
- `5` GPU page-touch kernel
- `6` VMA-wide `mbind` on first touch

For the dgemm micro-test, variants 2/3 look 3× faster than baseline, but on
the real workload they regress to ~62 s. Variant 4 looks best on the
micro-test (0.073 s) but is worst on `run2.sh` (~69 s). The micro-test is
**not** a reliable predictor — see `proxy/README.md`.

## Files modified

### `scilib-accel/utils/utils.c`

- Added the `SCILIB_MV` switch inside `move_numa` (selects one of 6
  implementations at runtime via env var).
- Added the **HBM-aware malloc interposer**:
  `malloc`, `calloc`, `realloc`, `free` are now exported by `scilib-dbi.so`.
  When `LD_PRELOAD`-ed, all glibc allocator calls flow through us. The
  interposer:
  - bootstraps safely (thread-local `in_dlsym` flag + 64 KB static buffer
    so `dlsym(RTLD_NEXT, "malloc")` cannot recurse);
  - parses `SCILIB_HBM_MALLOC_MB` lazily on first call;
  - for allocations ≥ threshold, calls `mbind(p, size, MPOL_BIND |
    MPOL_MF_MOVE, ...)` so the VMA's mempolicy is set to HBM and any
    currently-faulted pages are migrated.
- `move_numa2` is preserved as the `mbind`-based variant (reference).

### `scilib-accel/quick-test/run2.sh`

- Forwards `SCILIB_HBM_MALLOC_MB`, `SCILIB_MV`, and `SCILIB_DEBUG` to all
  ranks via `mpirun -x VAR` if they are set in the parent shell.
- Defaults (all vars unset) still produce the now-improved default behaviour.

### `scilib-accel/quick-test/exe.sh`

- Accepts `SCILIB_MV` as `$2` (legacy testing convenience).
- `SCILIB_DEBUG=1` is now a *default* (`: ${SCILIB_DEBUG:=1}`), letting a
  parent `SCILIB_DEBUG=0` actually take effect.

### `scilib-accel/proxy/` (new)

- `proxy.f90` — Fortran proxy that mirrors the MuST/LSMS pattern at the
  data-movement layer: 28-rank MPI launch, allocate / BLAS / deallocate
  cycle (fresh DRAM pages every iter), MuST-sized matrices.
- `Makefile`, `run_proxy.sh`, `README.md` — build and run.
- Wall time ~9 s; reproduces the *direction* of every variant we tested on
  `run2.sh` (HBM-malloc 64 MB wins, prefetch loses, etc.). Magnitudes are
  amplified because the proxy is migration-dominated.

## How to verify

From `quick-test/`:

```bash
./run2.sh                              # default-on  -> ~26.5 s
SCILIB_HBM_MALLOC_MB=0 ./run2.sh       # off          -> ~32 s
SCILIB_HBM_MALLOC_MB=128 ./run2.sh     # custom thr   -> ~26.8 s
SCILIB_MV=2 ./run2.sh                  # prefetch     -> ~62 s (regression check)
```

For pure CPU baseline (no `LD_PRELOAD`):

```bash
# 28 ranks × 2 OMP
mpirun --mca coll ^hcoll -np 28 -map-by node:PE=2 \
  bash -c 'export OMP_NUM_THREADS=2; numactl -m 0 ../mst2 < i_mst'
# 56 ranks × 1 OMP
mpirun --mca coll ^hcoll -np 56 -map-by node:PE=1 \
  bash -c 'export OMP_NUM_THREADS=1; numactl -m 0 ../mst2 < i_mst'
```

From `proxy/`:

```bash
cd proxy && make
./run_proxy.sh           # baseline,         max-rank ~9 s
./run_proxy.sh 3 0 64    # HBM-malloc 64 MB, max-rank ~1.1 s
./run_proxy.sh 3 2 0     # prefetch,         max-rank ~10 s
```

## Why HBM-aware malloc is the win

Profiling rank 0 of `run2.sh` (default `move_pages` path, before this work)
showed:

- 56 `move_numa` calls per rank, total 6.2 s.
- 9 calls accounted for 6.1 s (98 %); the other 47 were free
  (`move_pages` returns immediately when pages are already on target).
- All 9 slow calls had a pre-state of **0 pages on target** — they were
  genuine fresh allocations needing full migration.
- All 56 calls lived in **the same 1.1 GB heap VMA**.

The slowness wasn't redundant work to eliminate — it was the inherent cost
of migrating fresh DRAM pages to HBM. Trying to shortcut this via
`cudaMemPrefetchAsync` lost to driver per-call overhead under 28-rank
contention. Trying to skip migration (no-op / GPU-touch) lost the HBM
residency that made cuBLAS fast.

The malloc interposer moves the migration **earlier in time**, to
allocation rather than first-BLAS-use. Each large `allocate` ends with the
buffer already on HBM, so by the time a BLAS call arrives, `which_numa`
sees HBM and skips migration entirely. Per-rank `ztrsm.other` drops from
~5 s to ~0.025 s (−99.5 %); the app saves ~5.7 s end-to-end.

The 16–128 MB threshold range gives the same result (≈26.8 s). Below
~8 MB the interposer overhead on many small allocations starts to bite;
above ~256 MB we miss the 27 MB and 213 MB matrices. 64 MB chosen as
the default for robustness.

## What we tried that did **not** work

- Dropping `MPOL_MF_STRICT`, `MADV_HUGEPAGE` pre-mbind, multi-stream
  `cudaMemPrefetchAsync`, GPU write-touch kernel — no improvement on
  either test.
- VMA-wide `mbind` on first touch (`SCILIB_MV=6`): cuts ztrsm.other in
  half but pushes app to 57 s because CPU data on HBM is slower than CPU
  data on DRAM.
- `MALLOC_TRIM_THRESHOLD_=-1` / `MMAP_THRESHOLD_=4G`: no help (37 s).
- Range-based / smarter pointer cache: pre-state profiling proved there
  was no redundancy to skip.

## State of the repo

All work committed and pushed to `origin/main`:

- `b2aa6c5` HBM-aware malloc + `SCILIB_MV` + cleanup + proxy
- `3be0157` README MPS doc
- `85b17f2` 4 KB alignment default
- `6eba2d9` add `utils/gpu_migrate.{cu,h}`

Default behaviour changes:
- HBM-aware malloc on at **64 MB** threshold. Set `SCILIB_HBM_MALLOC_MB=0` to disable.
- Above-threshold allocations are aligned to **4 KB** via `posix_memalign`. Override with `SCILIB_HBM_MALLOC_ALIGN=N`.

## Follow-up wins (after the initial HBM-malloc work)

### NVIDIA MPS (~10 % extra, 26.5 s → 23.8 s)

Not a scilib-accel feature — it's an NVIDIA daemon (`nvidia-cuda-mps-control -d`)
that lets multiple CUDA processes share one context so their kernel launches can
overlap on the SMs instead of fully serialising at the GPU's single hardware
context. `quick-test/run2.sh` starts it automatically; `MPS=0 ./run2.sh` opts
out. Persistence mode must be on and the GPU compute mode must be `Default`.

The MPS active-thread-percentage knob (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`) was
swept across 3, 10, 25, 50, 100 — **100 % (the default) is best** for this
workload. Any restriction hurts.

### 4 KB alignment for above-threshold allocations (~3.9 %, 23.8 s → 22.8 s)

glibc returns 16-byte-aligned pointers; cuBLAS's vectorised loads issue extra
memory transactions when the base pointer isn't more strongly aligned. The
interposer now routes ≥threshold allocations through `posix_memalign` with
4 KB alignment. Sweep:

| alignment | mean app wall | Δ vs 16 B |
|---|---|---|
| 16 B (glibc default) | 23.7 s | — |
| 256 B   | 22.97 s | −0.73 s |
| **4 KB** | **22.83 s** | **−0.87 s** ← new default |
| 64 KB   | 22.92 s | −0.78 s |

Override via `SCILIB_HBM_MALLOC_ALIGN=N` (must be a power of two).

### preallocated cuBLAS workspace — no measurable effect

Adding `cublasSetWorkspace` with a 32 MiB pre-allocated HBM buffer in
`scilib_nvidia_init` did nothing on this workload (mean ±50 ms). cuBLAS's lazy
first-call workspace allocation amortises over the ~600 BLAS calls per rank, so
preallocating is invisible. Reverted; nothing committed.

### user-space THP — not possible without root

`madvise(MADV_HUGEPAGE)` is a no-op because the kernel's THP mode is `never`;
`MAP_HUGETLB` needs hugepages reserved in `/proc/sys/vm/nr_hugepages`; both
require sysadmin action. `cudaMalloc` provides hugepage-backed allocations but
**the resulting memory is GPU-only on Grace-Hopper** (the unified-address-space
only goes system → GPU, not GPU → system), so it can't be a drop-in
`malloc` replacement for Fortran-allocated arrays. `cudaMallocManaged` is
CPU+GPU-accessible but goes through the same managed/tracked code paths that
we already showed regress on the multi-rank workload (see `SCILIB_MV=2`/`3`).
**Conclusion: 22.8 s is the floor reachable without sysadmin or app-source
changes.**

## Final speedup ladder

```
Pure CPU baseline                                       126.7 s   1.00×
S3 alone (no HBM-malloc)                                 32.2 s   3.94×
S3 + HBM-malloc 64 MB (16-byte)                          26.5 s   4.78×
S3 + HBM-malloc + MPS (16-byte)                          23.8 s   5.33×
S3 + HBM-malloc + MPS + 4 KB align  (current default)    22.8 s   5.55×
```
