
#include <cublas_v2.h>
#include <cuda_runtime.h>

extern cublasStatus_t status;
extern cublasHandle_t handle;
extern cudaStream_t stream;

void nvidia_init();
void nvidia_fini();



