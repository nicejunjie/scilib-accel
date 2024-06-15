//  gcc -shared -fPIC -o libmymalloc.so mymalloc.c -ldl


#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <unistd.h>
#include <cuda_runtime.h>

static void* (*real_malloc)(size_t) = NULL;
static void* (*real_free)(void*) = NULL;

void* malloc(size_t size) {
    if (!real_malloc) {
        real_malloc = dlsym(RTLD_NEXT, "malloc");
        if (!real_malloc) {
            fprintf(stderr, "Error in dlsym: %s\n", dlerror());
            exit(EXIT_FAILURE);
        }
    }
    fprintf(stderr, "my malloc\n");
    void *ptr; 
//    ptr = real_malloc(size);
/*
    cudaError_t err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
*/
    ptr=malloc_managed(size);

    fprintf(stderr, "mymalloc(%zu) = %p (pid=%d)\n", size, ptr, getpid());

    return ptr;
}

void free(void* ptr) {
    if (!real_free) {
        real_free = dlsym(RTLD_NEXT, "free");
        if (!real_free) {
            fprintf(stderr, "Error in dlsym: %s\n", dlerror());
            exit(EXIT_FAILURE);
        }
    }
    fprintf(stderr, "my free\n");
//    real_free(ptr);
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaFree failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    fprintf(stderr, "myfree(%p) (pid=%d)\n", ptr, getpid());

    return;;
}

