#include <cuda_runtime.h>
#include <unistd.h>
#include <stdio.h>

// Touch one byte per page from GPU to trigger hardware page fault migration
__global__ void gpu_page_touch_kernel(char *ptr, size_t page_size, size_t num_pages) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < num_pages; i += stride) {
        volatile char v = ptr[i * page_size];
        (void)v;
    }
}

extern "C"
void gpu_migrate(void *ptr, size_t size, cudaStream_t stream) {
    size_t page_size = sysconf(_SC_PAGESIZE);
    size_t num_pages = (size + page_size - 1) / page_size;
    int threads = 256;
    int blocks = (num_pages + threads - 1) / threads;
    if (blocks > 2048) blocks = 2048;
    gpu_page_touch_kernel<<<blocks, threads, 0, stream>>>((char*)ptr, page_size, num_pages);
    cudaStreamSynchronize(stream);
}
