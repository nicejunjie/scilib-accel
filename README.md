# scilib-accel
Automatic GPU offload for scientific libraries. 

Only sgemm, dgemm, cgemm and zgemm are support at this point. 

Only NVIDIA GPU is supported but may support other GPU in the future. 

## Compile: 
`make` to make all 

[recommended] `make dbi` to make only the DBI-based version which works for both dynamically and statically linked BLAS, but needs FRIDA DBI library (downloaded automatically). 

`make dl` to make only the DLSYM-based version which only works with dynamically linked BLAS. This version has fewer features available.  


## Usage: 
load the chosen library before running your BLAS-heavy application.  

[recommended] `LD_PRELOAD=$PATH_TO_LIB/scilib-dbi.so` <br /> 
or  
`LD_PRELOAD=$PATH_TO_LIB/scilib-dl.so`   

Optionally use the following environmental variables to fine-tune: <br />
- `SCILIB_DEBUG=[0|1|2|3]` : 0 - default, no printouts, 1 - print timing, 2 -- print BLAS input arguments, 3 -- for developer diagnosis. <br />
- `SCILIB_MATRIX_OFFLOAD_SIZE=[size]` : size=(mnk)^(1/3), default is 500, the size above which GPU offload will occur.  <br />
- `SCILIB_THPOFF=[0|1]` : 0 - default, use system default THP setting, 1 -- turn off THP.  <br />
- `SCILIB_OFFLOAD_MODE=[1|2|3]`: different data movement strategies.  <br/>
  - 1: perform cudaMemCpy to/from GPU for every cuBLAS call;  (available on any GPU)  
  - 2: use unified memory access without explicit data movement;  (only available on Grace-Hopper)
  - 3: (default) apply First GPU Use Policy, data is migrated to GPU HBM upon the first use of cuBLAS and stay resident on HBM. (Only available on Grace-Hopper)

## Known issues: 
Bugs from the UCX side were observed, UCX driver somehow interferes with memory pages and causes issue using NUMA 1 (the HBM). Before NVIDIA fixes the bug, UCX has to be turned off: <br /> 
```bash
export OMP_NUM_THREADS=$nt
mpirun --mca pml ^ucx --mca btl self,vader,tcp -n $nrank -map-by node:PE=$nt $EXE
```
or 
```bash
export OMP_NUM_THREADS=$nt
export OMPI_MCA_pml="^ucx"
export OMPI_MCA_btl="self,vader,tcp"
export OMPI_MCA_coll="^hcoll"
export OMPI_MCA_btl_tcp_if_exclude="lo"

mpirun -n $nrank -map-by node:PE=$nt $EXE
```

## Reference: 
[Automatic BLAS Offloading on Unified Memory Architecture: A Study on NVIDIA Grace-Hopper](https://arxiv.org/abs/2404.13195)
