#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#define CUDA_CHECK(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(result)); \
        exit(result); \
    } \
} while(0)

// Memory pool structure
typedef struct MemoryPool {
    size_t size;
    size_t offset;
    void* data;
} MemoryPool;

// Function to create a memory pool
MemoryPool createMemoryPool(size_t size) {
    MemoryPool pool;
    pool.size = size;
    pool.offset = 0;

    // Allocate memory for the pool
    CUDA_CHECK(cudaMalloc(&pool.data, size));

    return pool;
}

// Function to destroy a memory pool
void destroyMemoryPool(MemoryPool* pool) {
    CUDA_CHECK(cudaFree(pool->data));
}

// Function to allocate memory from the memory pool
void* allocateFromPool(MemoryPool* pool, size_t size) {
    int newsize=pool->offset + size - pool->size;
    if (newsize > 0) {
        fprintf(stderr, "Memory pool exhausted; %.0f MB more is needed\n", (double)newsize/1024.0/1024.0);
        return NULL;
    }

    void* allocatedMemory = (char*)pool->data + pool->offset;
    pool->offset += size;

    return allocatedMemory;
}

// Function to deallocate memory to the memory pool (not used in this example)
void deallocateToPool(MemoryPool* pool, void* ptr) {
    // Deallocation is not explicitly needed in this example,
    // as the memory pool uses a simple linear allocation strategy.
    // The entire pool can be reused when necessary.
}
