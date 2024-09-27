#!/bin/bash

ml purge
ml use /scratch/07893/junjieli/soft/nvhpc/24.7/modulefiles
ml load  nvhpc-hpcx-cuda12/24.7

CPPFLAGS="-DMOVE_NUMA "
#-DMEM_ALIGN  -DMOVE_NUMA -DMEM_ADVISE

#MEMMODE="-gpu=mem:separate"
 MEMMODE="-gpu=mem:unified"
#MEMMODE="-gpu=mem:managed"
CUDA_HOME=/scratch/07893/junjieli/soft/nvhpc/24.7/Linux_aarch64/24.7
#TMPLIB=/scratch/07893/junjieli/soft/nvhpc/24.7/Linux_aarch64/24.7/compilers/lib/libaccstub.so
#TMPLIB=/home1/07893/junjieli/scilib-accel/jemalloc/lib/libjemalloc.a
nvc $CPPFLAGS $MEMMODE dgemm-cmp3.c -mp -Mnvpl -cuda -I${CUDA_HOME}/cuda/include -I${CUDA_HOME}/math_libs/include/ -L${CUDA_HOME}/math_libs/lib64 -lcublas -lnuma  $TMPLIB #-nvmalloc

export OMP_NUM_THREADS=72


# ./a.out M N K
# C = A * B, 
# matrix dimensions:  A(M,K), B(K,N), C(M,N) 

#export SCILIB_DEBUG=0
#export SCILIB_OFFLOAD_MODE=3
#export LD_PRELOAD=~/scilib-accel/scilib-dbi.so

#./a.out  2048 2048 2048 10

#export LD_PRELOAD=/home1/07893/junjieli/scilib-accel/jemalloc/lib/libjemalloc.so 
 ./a.out 32 2400 93536 5
#./a.out 1000 1000 1000 5
