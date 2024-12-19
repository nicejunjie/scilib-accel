
#include <stdio.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include "global.h"

cublasHandle_t scilib_cublas_handle;
cusolverDnHandle_t scilib_cusolverDn_handle;
cudaStream_t scilib_cuda_stream;


void scilib_nvidia_init(){

/*  CUBLAS  */
    cublasStatus_t cublas_status;
    cublas_status = cublasCreate(&scilib_cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        exit(0);
        return;
    }

/*  CUSOLVER  */
    cusolverStatus_t cusolver_status;
    cusolver_status = cusolverDnCreate(&scilib_cusolverDn_handle);
    if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "CUSOLVER initialization failed\n");
        exit(0);
        return;
    }

    if(scilib_offload_mode == 1) {
        cudaStreamCreate(&scilib_cuda_stream);
        cublasSetStream(scilib_cublas_handle, scilib_cuda_stream);
        cusolverDnSetStream(scilib_cusolverDn_handle, scilib_cuda_stream);
    }

    return;
}


void scilib_nvidia_fini(){

    /*  CUBLAS  */
    cublasDestroy(scilib_cublas_handle);
    
    /*  CUSOLVER  */
    cusolverDnDestroy(scilib_cusolverDn_handle);

    if(scilib_offload_mode == 1)
        cudaStreamDestroy(scilib_cuda_stream);

    return;
}


