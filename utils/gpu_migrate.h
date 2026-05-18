#ifndef GPU_MIGRATE_H
#define GPU_MIGRATE_H

#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void gpu_migrate(void *ptr, size_t size, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
