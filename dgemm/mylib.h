#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


#ifdef GPU_COPY
#define CUDA_ASYNC
#endif
