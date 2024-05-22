#!/bin/bash 

NVHOME=$(dirname $(dirname $(dirname $(which pgf90))))
CUDAHOME=$NVHOME/cuda

export LD_LIBRARY_PATH=$CUDAHOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDAHOME/lib64:$LIBRARY_PATH
export PATH=$CUDAHOME/bin:$PATH
export CPATH=$CUDAHOME/include:$CPATH

pgcc -shared -fPIC -o libmymalloc.so mymalloc.c -ldl -gpu=managed -lcudart

pgf90 -O2 -lblas test_dgemm.f90 -o dgemm.x

LD_PRELOAD=./libmymalloc.so echo 4000 4000 4000 10 |./dgemm.x
