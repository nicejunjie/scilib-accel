#!/bin/bash

NV=24.3
NVHOME=/scratch1/07893/junjieli/grace-hopper/junjieli/soft/nvhpc/24.3/Linux_aarch64/24.3
CUBLAS=$NVHOME/math_libs/lib64/libcublas.so
echo $CUBLAS
CURT=$NVHOME/cuda/lib64/libcudart.so
CUINCLUDE=$NVHOME/cuda/include
 
echo $CUINCLUDE

LIBS="$CUBLAS $CURT"

export CPATH=$CUINCLUDE:$CPATH

pgcc -I$CUINCLUDE tmp.c $LIBS

