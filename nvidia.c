
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

cublasStatus_t status;
cublasHandle_t handle;
#ifdef GPUCOPY
cudaStream_t stream;
#endif


void nvidia_init(){

/*  CUBLAS  */
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        return;
    }

#ifdef GPUCOPY
    cudaStreamCreate(&stream);
#endif 

    return;
}


void nvidia_fini(){

/*  CUBLAS  */
    cublasDestroy(handle);
#ifdef GPUCOPY
    cudaStreamDestroy(stream);
#endif

    return;
}


