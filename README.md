# scilib-accel
automatic GPU offload for scientific libraries 

# compile: 
`make` to make all
`make dbi` to make only the DBI-based version which works for both dynamically and statically linked BLAS, but needs FRIDA DBI library (downloaded automatically).
`make dl` to make only the DLSYM-based version which only works dynamically linked BLAS. 

# usage: 
load the chosen library before running your BLAS heavy application. 
`LD_PRELOAD=$PATH_TO_LIB/scilib-dbi.so`
or  
`LD_PRELOAD=$PATH_TO_LIB/scilib-dl.so` 

