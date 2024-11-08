# SCILIB-Accel
Automatic GPU offload for scientific libraries (hopefully not just BLAS).  Designed for NVIDIA Grace-Hopper. 

All level-3 BLAS subroutines are supported now! <br />
*gemm, *symm, *trmm, *trsm, *syrk, *syr2k, *hemm, *herk, *her2k.

fftw support in progress.

Only NVIDIA GPU is supported but may support other GPU in the future. 

For a fully functional BLAS/LAPACK/ScaLAPACK profiler, please refer to my other tool: 
[SCILIB-Prof](https://github.com/nicejunjie/scilib-prof/tree/main )


# More about SCILIB-Accel auto offload approach: 
BLAS auto offload isn't new, since Cray LIBSCI, IBM ESSL, NVIDIA NVBLAS all attempt to do offload. 
However, these libraries suffer huge cost of data transfer by copying matrices to/from GPU for every BLAS call.  
Additionally, NVBLAS is heavily over-engineered and has excessive implementation overhead. 
Therefore, tools are never practically useful. 

Recognizing common use patterns of BLAS calls, SCILIB-accel introduces a first-touch type of data management strategy (S3 below) optimized for NVIDIA Grace-Hopper,
 data movement in many practical use cases is minimum. 

To my knowledge, this is the first tool that allows  performant BLAS auto-offload on GPU for real HPC applications. 

## Compile: 
`make` to make all 

[recommended] `make dbi` to make only the DBI-based version which works for both dynamically and statically linked BLAS, but needs FRIDA DBI library (downloads automatically).  
DBI interferes with many profilers, so if you'd like to attach a profiler, use the DLSYM-based approach below. 

`make dl` to make only the DLSYM-based version which only works with dynamically linked BLAS. This version has no dependency on the external DBI library but also has fewer features available (key offload features are not affected).  


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
  - S1: perform cudaMemCpy to/from GPU for every cuBLAS call;  (available on any GPU)  
  - S2: use unified memory access without explicit data movement;  (only available on Grace-Hopper)
  - S3: (default) apply GPU First Use policy, data is migrated to GPU HBM upon the first use of cuBLAS and stay resident on HBM. 
        This policy is very similiar to OpenMP First Touch, with the assumption that the CPU access of the migrated matrices on HBM
         are relatively trivial compared to the amount of GPU local access. 
        (Only available on Grace-Hopper)

## Known issues: 
For using openmpi in NVHPC, bugs from the UCX side were observed and hcoll should be disabled. <br /> 

```bash
export OMP_NUM_THREADS=$nt
export OMPI_MCA_coll="^hcoll"
mpirun -n $nrank -map-by node:PE=$nt $EXE
```
<!-- export OMPI_MCA_btl_tcp_if_exclude="lo" -->
<!-- export OMPI_MCA_btl="self,vader,tcp" --> 
<!-- export OMPI_MCA_pml="^ucx" -->
<!-- export OMPI_MCA_btl="self,vader" -->



## Latest test data:  
**PARSEC  <br />**
Real-space Density Functional Theory code https://real-space.org.   
These are single node runs for a system with about 2000 Si atoms. 
| Method | App Total Runtime | DGEMM Time | Data Movement | Notes |
|--------|---------------------------|------------|---------------|-------|
| CPU, single Grace | 776.5 | 608s | 0 | |
| SCILIB-Accel S1: data copy | 425.7s | 12.4s | 220.7s | |
| SCILIB-Accel S2: pin on HBM | 299.6s | 28.5s | 0 | |
| SCILIB-Accel S2: -gpu=unified * | 246.8s | 56.6s | N/A | 64k page |
| SCILIB-Accel S3: GPU First Use | 220.3s | 29.1s | 1.3s | Matrix reuse: 570 | 

\* require both PARSEC and SCILIB-Accel to be recompiled with -gpu=unified, and only works well with 64k page due to CUDA bugs in 4k.  
-gpu=managed is another related flag enabling managed memory, it performs significantly worse. 
Grace-Hopper also supports counter-based page migration, it is the least performing one. 


**MuST <br />**
Multiple Scattering Theory code for first principle calculations https://github.com/mstsuite/MuST  

The code has a CUDA port, but auto-offload is 2x faster than the native CUDA code. 


<!-- This test case here is a LSMS run for 56-atom alloy system on single node. -->
<!-- | Method | App Total Runtime | ZGEMM+ZTRSM Time  | Data Movement | Notes |-->
<!-- |--------|---------------------------|------------|---------------|-------|-->
<!-- | CPU, single Grace | 124s | 82.5s + 35.2s | 0 | |-->
<!-- | Native GPU (cuSolver) | 57.4s | N/A | N/A | |-->
<!-- | SCILIB-Accel S1: data copy | 31.5s | 11.7s + 1.4s | 13.6s | |-->
<!-- | SCILIB-Accel S3: GPU First Use | 30.7 | 15.9s + 3.8s | 3.6s | matrix reuse: 70 |-->  

Test case: LSMS run for 5600-atom alloy system. 
This workload can perfectly scale from 25 nodes to 150 nodes, GH vs GG speedup 2.8~3.2x using S3: GPU First Use.  
| Method | App Total Runtime | ZGEMM+ZTRSM Time  | Data Movement | Notes | 
|--------|---------------------------|------------|---------------|-------|
| 150 GG CPU nodes | 997s | - | 0 | |
| 150 GH GPU nodes, Native GPU (cuSolver) | 673s | - | 0 | |  
| 150 GH GPU nodes, SCILIB-Accel S1 | 435s | 152s+17s | ~100s | - |  
| 150 GH GPU nodes, SCILIB-Accel S3 | 357s | 184s+35s | 3.3s | matrix reuse 780 |  


**HPL (binary from NVIDIA's HPC container)**
| HPL Method | Rmax (TFlops) | t_dgemm (s) | t_data (s) | Notes |
|------------|---------------|-------------|------------|-------|
| CPU, single Grace | 2.8 | 188.8 | 0 | |
| SCILIB-Accel S2: pin on HBM | 25.7 | 11.6 | 0 | |
| SCILIB-Accel S3: GPU First Use | 11.3 | 14.2 | 9.2 | matrix reuse: 6 |
| Native GPU | 51.7 | - | - | |

## Reference:  
For more details, please see this [presentation](https://github.com/nicejunjie/scilib-accel/blob/main/presentation/BLAS-auto-offload.pdf)

This paper summarizes the early developments. The current performance is much better than it was outlined in the paper. <br />
[Automatic BLAS Offloading on Unified Memory Architecture: A Study on NVIDIA Grace-Hopper](https://arxiv.org/abs/2404.13195)
