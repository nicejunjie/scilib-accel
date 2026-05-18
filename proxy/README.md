# proxy — fast benchmark for `move_numa` / HBM-malloc work

## Why it exists

`test_dgemm.x` (single process, one buffer reused 5 times) does **not**
predict `quick-test/run2.sh`. Two examples we've seen:

| variant | `test_dgemm` says | `run2.sh` actually | proxy here says |
|---|---|---|---|
| `cudaMemPrefetchAsync` | 3× faster than `move_pages` | 2× **slower** | slower |
| HBM-aware malloc (64 MB threshold) | hard to tell | 17 % faster | 87 % faster |

`run2.sh` is the source of truth, but it's slow (~32 s/run × N reps). The
proxy fires the **same shape** of workload — 28 MPI ranks, allocate /
BLAS / free with no buffer reuse, MuST-sized matrices — in ~9 s per run,
so you can iterate on `utils/utils.c` or the malloc interposer without
waiting half a minute per try.

## Build

    cd proxy
    make

Needs `pgf90`/NVPL (same toolchain as the rest of the library).

## Run

    ./run_proxy.sh [niter] [SCILIB_MV] [SCILIB_HBM_MALLOC_MB]

- `niter`: how many (zgemm + ztrsm) outer iterations per rank (default 3).
- `SCILIB_MV`: `move_numa` variant 0..6 (default 0 = `move_pages`).
- `SCILIB_HBM_MALLOC_MB`: HBM-aware-malloc threshold in MB (default 0 = off).

Examples:

    ./run_proxy.sh                  # baseline
    ./run_proxy.sh 3 2              # try cudaMemPrefetchAsync
    ./run_proxy.sh 3 0 64           # HBM-aware malloc, 64 MB threshold

Override the launch with `RANKS=` / `NT=` if you want a smaller/larger
contention pattern; default matches `run2.sh` (28 ranks × OMP=2).

## How to read it

The script prints two numbers:

- `mpirun wall` — wall time of the whole mpirun invocation, including
  start-up. Noisy; only useful as a ceiling.
- `max-rank` — the slowest per-rank `proxy rank time` reported by the
  Fortran program. **This is the figure to compare against.**

The proxy is migration-dominated (very little compute), so it **amplifies**
the magnitude of any change to `move_numa`. Trust the *sign* and the
*ordering* of variants; for final percentages, validate on `run2.sh` and
read the "Job total including IO" line of
`quick-test/o_n0000000_CoCrFeMnNi`.

## What it reproduces (good)

- 28-rank GPU contention.
- `malloc` → BLAS → `free` cycle with fresh DRAM pages each iteration
  (matches the pattern in MuST/LSMS at the heap level).
- Same buffer-size families as profiled in MuST: 26 MB, 204 MB, 241 MB,
  487 MB.

## What it does **not** reproduce (don't trust it for these)

- The overall app's compute/migration ratio. Real MuST is ~30 % migration;
  proxy is ~80 %. Hence the magnitude inflation.
- MPI all-reduces, file I/O, and other serial sections of MuST.
- Long-tail effects from running for many minutes (cache state, GPU
  thermal behaviour, etc.).
