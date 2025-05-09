#ifndef SCILIB_PTHREAD_WRAP_H
#define SCILIB_PTHREAD_WRAP_H

#include <pthread.h>
#include <cuda_runtime.h>

// Function pointer for the original pthread_create
extern int (*real_pthread_create)(pthread_t *__restrict,
                                   const pthread_attr_t *__restrict,
                                   void *(*__start_routine)(void *),
                                   void *__restrict);

// Our wrapper/interceptor for pthread_create
// For LD_PRELOAD, this function will be named "pthread_create"
// For Frida, this will be the replacement function
int scilib_pthread_create_wrapper(pthread_t *__restrict thread,
                                  const pthread_attr_t *__restrict attr,
                                  void *(*start_routine)(void *),
                                  void *__restrict arg);

// Gets the CUDA stream assigned to the current thread
cudaStream_t scilib_get_current_thread_stream(void);

// To be called from scilib_elf_init to setup interception (especially for Frida)
// and to assign a stream to the main thread.
void scilib_pthread_wrap_init(void);

#endif // SCILIB_PTHREAD_WRAP_H