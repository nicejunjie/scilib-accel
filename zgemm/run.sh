 


CUBLAS=/home/nvidia/junjieli/soft/nvhpc/23.11/Linux_aarch64/23.11/math_libs/lib64/libcublas.so
CURT=/home/nvidia/junjieli/soft/nvhpc/23.11/Linux_aarch64/23.11/cuda/lib64/libcudart.so
CUINCLUDE=/home/nvidia/junjieli/soft/nvhpc/23.7/src/nvhpc_2023_237_Linux_aarch64_cuda_multi/install_components/Linux_aarch64/23.7/cuda/12.2/include/
CUINCLUDE=/home/nvidia/junjieli/soft/nvhpc/23.11/Linux_aarch64/23.11/cuda/include

#COPY="-DGPUCOPY -DDEBUG -DCUDA_MEM_POOL"
#COPY="-DGPUCOPY -DCUDA_MEM_POOL"
#COPY="-DAUTO_NUMA"
 COPY="-DGPUCOPY"


CFLAGS=" -O2 -lnuma -mp -gpu=unified"
EXTRA_FLAGS="--diag_suppress incompatible_assignment_operands --diag_suppress set_but_not_used --diag_suppress incompatible_param"
FLAGS="$CFLAGS $EXTRA_FLAGS"

CC=mpicc

$CC -c -g -fPIC mysecond.c -o mysecond.o  $FLAGS
$CC $COPY -c -g   -fPIC mylib.c  -o mylib.o  -I$CUINCLUDE -traceback $FLAGS
$CC -shared -g  -o mylib.so mylib.o mysecond.o $CUBLAS $CURT -traceback $FLAGS 

pgfortran -g -mp  -lblas -O2 -Minfo=all test_zgemm.f90 mysecond.o


#export PGI_TERM=trace #debug trace signal abort 


#M=2233
#N=4516
#K=2234
#M=2233 
#N=2283 
#K=4467

M=1754
N=32
K=32

ni=3

echo "-------------------------"
echo Matrix Size: $M $N $K
echo "-------------------------"
export OMP_NUM_THREADS=72
echo ""
 ./a.out $M $N $K $ni
echo ""
LD_PRELOAD=./mylib.so ./a.out $M $N $K $ni
echo ""
#LD_PRELOAD=./mylib.so  numactl -m 1 ./a.out $M $N $K $ni
