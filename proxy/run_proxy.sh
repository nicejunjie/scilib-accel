#!/bin/bash
# Run the proxy benchmark under the same MPI launch pattern as quick-test/run2.sh.
#
# Usage: ./run_proxy.sh [niter] [SCILIB_MV] [SCILIB_HBM_MALLOC_MB]
#   niter:                # of (zgemm+ztrsm) outer iters per rank   (default 3)
#   SCILIB_MV:            move_numa variant 0..6                    (default 0)
#   SCILIB_HBM_MALLOC_MB: HBM-malloc threshold MB (0 = off)          (default 0)
#
# Examples:
#   ./run_proxy.sh                  # baseline
#   ./run_proxy.sh 3 0 64           # add HBM-aware malloc, threshold 64 MB
#   ./run_proxy.sh 3 2              # try cudaMemPrefetchAsync (variant 2)

niter=${1:-3}
SCILIB_MV=${2:-0}
SCILIB_HBM_MALLOC_MB=${3:-0}

ranks=${RANKS:-28}
nt=${NT:-2}

here=$(cd "$(dirname "$0")" && pwd)
lib=$here/../scilib-dbi.so

[ -x "$here/proxy.x" ] || { echo "build proxy.x first (cd proxy; make)"; exit 1; }
[ -e "$lib"       ] || { echo "scilib-dbi.so not found at $lib"; exit 1; }

cat > /tmp/proxy_exe.sh <<EOF
#!/bin/bash
export OMP_NUM_THREADS=\$1
[ -n "\$2" ] && export SCILIB_MV=\$2
[ -n "\$3" ] && export SCILIB_HBM_MALLOC_MB=\$3
export LD_PRELOAD=$lib
exec numactl -m 0 $here/proxy.x <<< "$niter"
EOF
chmod +x /tmp/proxy_exe.sh

t_start=$(date +%s.%N)
out=$(mpirun --mca coll ^hcoll -np $ranks -map-by node:PE=$nt \
        /tmp/proxy_exe.sh $nt $SCILIB_MV $SCILIB_HBM_MALLOC_MB 2>&1)
t_end=$(date +%s.%N)
wall=$(awk -v a=$t_start -v b=$t_end 'BEGIN{printf "%.3f", b-a}')

# max-over-ranks of the proxy-rank time
max_rank=$(echo "$out" | awk '/proxy rank time/ {if ($NF+0 > m) m = $NF+0} END{printf "%.3f", m}')

printf "niter=%s  SCILIB_MV=%s  SCILIB_HBM_MALLOC_MB=%s  ::  mpirun wall=%.3fs  max-rank=%.3fs\n" \
    "$niter" "$SCILIB_MV" "$SCILIB_HBM_MALLOC_MB" "$wall" "$max_rank"
