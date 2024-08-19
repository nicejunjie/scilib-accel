
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "global.h"

cublasStatus_t status;
cublasHandle_t handle;
cudaStream_t stream;


void scilib_nvidia_init(){

/*  CUBLAS  */
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        return;
    }

if(scilib_offload_mode == 1)
    cudaStreamCreate(&stream);
    return;
}


void scilib_nvidia_fini(){

/*  CUBLAS  */
    cublasDestroy(handle);
if(scilib_offload_mode == 1)
    cudaStreamDestroy(stream);
    return;
}


