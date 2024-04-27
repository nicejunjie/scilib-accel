

#define GiB 1024*1024*1024;
#define MiB 1024*1024;
#define KiB 1024;

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

