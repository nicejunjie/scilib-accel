#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

int main() {
    // Define array size
    const int N = 10;

    // Host array
    double *h_array = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        h_array[i] = 1.0*i;
    }

    // Device array
    double *d_array;
    CUDA_CHECK(cudaMalloc((void **)&d_array, N * sizeof(double)));

    // Copy from host to device asynchronously
    int lda=30;
    for (int i=0; i< N; i++)
       CUDA_CHECK(cudaMemcpyAsync(d_array+i*lda, h_array+i*lda, 1 * sizeof(double), cudaMemcpyHostToDevice, 0));

    // Wait for asynchronous copy to complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free allocated memory
    free(h_array);
    CUDA_CHECK(cudaFree(d_array));

    return 0;
}

