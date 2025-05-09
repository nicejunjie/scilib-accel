
#include <stdio.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include "global.h"

cublasHandle_t scilib_cublas_handle;
cusolverDnHandle_t scilib_cusolverDn_handle;
cudaStream_t *scilib_cuda_streams = NULL;

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

    scilib_cuda_streams = (cudaStream_t *)malloc(scilib_num_cuda_streams * sizeof(cudaStream_t));
    if (!scilib_cuda_streams) {
        fprintf(stderr, "SCILIB: Failed to allocate memory for CUDA streams\n");
        exit(0);
    }
    for (int i = 0; i < scilib_num_cuda_streams; ++i) {
        cudaError_t err = cudaStreamCreate(&scilib_cuda_streams[i]);
        if (err != cudaSuccess) {
            fprintf(stderr, "SCILIB: Failed to create CUDA stream %d: %s\n", i, cudaGetErrorString(err));
            // Handle error, maybe exit or try to continue with fewer streams
            exit(0);
        }
    }

    // if(scilib_offload_mode == 1) {
    //     cudaStreamCreate(&scilib_cuda_stream);
    //     cublasSetStream(scilib_cublas_handle, scilib_cuda_stream);
    //     cusolverDnSetStream(scilib_cusolverDn_handle, scilib_cuda_stream);
    // }

    return;
}


void scilib_nvidia_fini(){

    if (scilib_cuda_streams) {
        for (int i = 0; i < scilib_num_cuda_streams; ++i) {
            if (scilib_cuda_streams[i]) {
                cudaStreamDestroy(scilib_cuda_streams[i]);
            }
        }
        free(scilib_cuda_streams);
        scilib_cuda_streams = NULL;
    }

    /*  CUBLAS  */
    cublasDestroy(scilib_cublas_handle);
    
    /*  CUSOLVER  */
    cusolverDnDestroy(scilib_cusolverDn_handle);

    // if(scilib_offload_mode == 1)
    //     cudaStreamDestroy(scilib_cuda_stream);

    return;
}


