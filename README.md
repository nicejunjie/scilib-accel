# scilib-accel
Automatic GPU offload for scientific libraries. 

Only sgemm, dgemm, cgemm and zgemm are support at this point. 

Only NVIDIA GPU is supported but may support other GPU in the future. 

## Compile: 
`make` to make all 

[suggested] `make dbi` to make only the DBI-based version which works for both dynamically and statically linked BLAS, but needs FRIDA DBI library (downloaded automatically). 

`make dl` to make only the DLSYM-based version which only works dynamically linked BLAS. This version has less feature available.  

Change `CPPFLAGS` in Makefile from `-DGPUCOPY`, none (which indicates zero copy), or `-DAUTO_NUMA` to select different data management strategy.  `-DAUTO_NUMA` will work best for most cases on Grace-Hopper, while non UMA GPUs only supports "-DGPUCOPY". 

## Usage: 
load the chosen library before running your BLAS heavy application.  

`LD_PRELOAD=$PATH_TO_LIB/scilib-dbi.so` <br /> 
or  
`LD_PRELOAD=$PATH_TO_LIB/scilib-dl.so`   

Optionally use the following environmental variables to fine-tune: <br />
- `SCILIB_DEBUG=[0|1|2]` : 0 - default, 1 - print timing, 2 -- print BLAS input arguments. <br />
- `SCILIB_MATRIX_OFFLOAD_SIZE=[size]` : size=(mnk)^(1/3), the size above which GPU offload will occur. <br />
- `SCILIB_THPOFF=[0|1]` : 0 - default, use system default THP setting, 1 -- turn off THP.  <br />
- `SCILIB_OFFLOAD_MODE=[1|2|3]`: different data movement strategies.  <br/>
  - 1: perform cudaMemCpy to/from GPU for every cuBLAS call;  (available on any GPU)  
  - 2: use unified memory access without explicit data movement;  (only available on Grace-Hopper)
  - 3: (default) apply First GPU Use Policy, data is migrated to GPU HBM upon the first use of cuBLAS and stay resident on HBM. (Only available on Grace-Hopper)


## Reference: 
[Automatic BLAS Offloading on Unified Memory Architecture: A Study on NVIDIA Grace-Hopper](https://arxiv.org/abs/2404.13195)
