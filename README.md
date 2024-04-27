# scilib-accel
Automatic GPU offload for scientific libraries. 

Only dgemm and zgemm are support at this point. 

## Compile: 
`make` to make all 

`make dbi` to make only the DBI-based version which works for both dynamically and statically linked BLAS, but needs FRIDA DBI library (downloaded automatically). 

`make dl` to make only the DLSYM-based version which only works dynamically linked BLAS.  

Change `CPPFLAGS` in Makefile from `-DGPUCOPY`, none (which indicates zero copy), or `-DAUTO_NUMA` to select different data management strategy.  `-DAUTO_NUMA` will work best for most cases on Grace-Hopper, while non UMA GPUs only supports "-DGPUCOPY". 

## Usage: 
load the chosen library before running your BLAS heavy application.  

`LD_PRELOAD=$PATH_TO_LIB/scilib-dbi.so` <br /> 
or  
`LD_PRELOAD=$PATH_TO_LIB/scilib-dl.so`  

## Reference: 
[Automatic BLAS Offloading on Unified Memory Architecture: A Study on NVIDIA Grace-Hopper](https://arxiv.org/abs/2404.13195)
