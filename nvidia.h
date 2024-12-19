
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>


extern cublasHandle_t scilib_cublas_handle;
extern cusolverDnHandle_t scilib_cusolverdn_handle;
extern cudaStream_t scilib_cuda_stream;

void scilib_nvidia_init();
void scilib_nvidia_fini();




#define CUDA_CHECK(call)                                                  \
do {                                                                      \
    cudaError_t error = call;                                             \
    if (error != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error: %s:%d, ", __FILE__, __LINE__);       \
        fprintf(stderr, "code: %d, reason: %s\n", error,                  \
                cudaGetErrorString(error));                               \
        exit(1);                                                          \
    }                                                                     \
} while (0)


#define CUBLAS_CHECK(call)                                              \
do {                                                                \
    cublasStatus_t status = call;                                   \
    if (status != CUBLAS_STATUS_SUCCESS) {                          \
        const char* error_str = cublasGetStatusString(status);      \
        fprintf(stderr, "CUBLAS error: %s in %s:%d\n",              \
                error_str, __FILE__, __LINE__);                     \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while (0)


#define CUSOLVER_CHECK(call)                                            \
do {                                                                    \
    cusolverStatus_t status = call;                                     \
    if (status != CUSOLVER_STATUS_SUCCESS) {                           \
        fprintf(stderr, "CUSOLVER error: %d in %s:%d\n",               \
                status, __FILE__, __LINE__);                           \
        exit(EXIT_FAILURE);                                            \
    }                                                                   \
} while (0)
