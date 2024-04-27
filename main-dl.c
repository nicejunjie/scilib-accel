#define _GNU_SOURCE

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include "utils.h"
#include "init.h"

//to disable THP
#include <sys/prctl.h> 


freplace farray[] = {
  INIT_FARRAY
};
int fsize = sizeof(farray) / sizeof(farray[0]);

cublasStatus_t status;
cublasHandle_t handle;

#ifdef CUDA_ASYNC
cudaStream_t stream;
#endif


void my_init(){

  fprintf(stderr, "SCILIB-accel DL");

// disable THP for auto-page migration
#ifdef AUTO_NUMA
     prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
#endif

/*  CUBLAS  */
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        return;
    }

#ifdef GPUCOPY
    cudaStreamCreate(&stream);
#endif

// register functions
  for( int i=0; i< fsize; i++) {
     farray[i].fptr= dlsym(RTLD_NEXT, farray[i].f0);
  }

  return;
}


void my_fini(){
    cublasDestroy(handle);
#ifdef GPUCOPY
    cudaStreamDestroy(stream);
#endif
/*
   if(mtime_total>0.000001){
              fprintf(stderr,"dgemm time total= %.6f, data=%.6f, compute=%.6f\n", mtime_total,mtime_dmove,mtime_comput);
#ifdef GPU_COPY
              fprintf(stderr, "data vol (GB): %.6f, copy speed GB/s: %.6f\n", mvol_dmove, mvol_dmove/mtime_dmove);
#endif
   }
*/

    fflush(stderr);
    fflush(stdout);

    return;
}

  __attribute__((section(".init_array"))) void *__init = my_init;
  __attribute__((section(".fini_array"))) void *__fini = my_fini;


