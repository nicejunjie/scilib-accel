#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include "scilib_pthread_wrap.h"
#include "nvidia.h" // For scilib_cuda_streams, scilib_num_cuda_streams
#include "global.h" // For any debug macros if needed

int (*real_pthread_create)(pthread_t *__restrict,
                           const pthread_attr_t *__restrict,
                           void *(*__start_routine)(void *),
                           void *__restrict) = NULL;

// Thread-Local Storage for the stream
__thread cudaStream_t current_thread_stream_tls = (cudaStream_t)-2; // Uninitialized state

static pthread_mutex_t stream_assignment_lock = PTHREAD_MUTEX_INITIALIZER;
static int next_stream_index = 0;

// Internal function to assign a stream
static void assign_stream_to_current_thread() {
    pthread_mutex_lock(&stream_assignment_lock);
    if (scilib_num_cuda_streams > 0 && scilib_cuda_streams != NULL) {
        current_thread_stream_tls = scilib_cuda_streams[next_stream_index];
        next_stream_index = (next_stream_index + 1) % scilib_num_cuda_streams;
    } else {
        // Fallback to default stream (0) if streams aren't initialized
        // or if scilib_num_cuda_streams is 0.
        current_thread_stream_tls = (cudaStream_t)0;
    }
    pthread_mutex_unlock(&stream_assignment_lock);
}

cudaStream_t scilib_get_current_thread_stream(void) {
    if (current_thread_stream_tls == (cudaStream_t)-2) { // Check for uninitialized state
        assign_stream_to_current_thread();
    }
    return current_thread_stream_tls;
}

typedef struct {
    void *(*user_start_routine)(void *);
    void *user_arg;
} thread_starter_args_t;

static void *thread_starter_wrapper(void *arg) {
    thread_starter_args_t *wrapper_args = (thread_starter_args_t *)arg;
    void *(*actual_start_routine)(void *) = wrapper_args->user_start_routine;
    void *actual_arg = wrapper_args->user_arg;

    free(wrapper_args); // Clean up the temporary args structure

    // Assign a stream to this new thread when it starts
    assign_stream_to_current_thread();

    return actual_start_routine(actual_arg);
}

// For LD_PRELOAD, rename this function to "pthread_create"
// For Frida, keep this name and hook "pthread_create" to point to this.
int scilib_pthread_create_wrapper(pthread_t *__restrict thread,
                                  const pthread_attr_t *__restrict attr,
                                  void *(*start_routine)(void *),
                                  void *__restrict arg) {
    if (!real_pthread_create) {
        // This dlsym is essential for LD_PRELOAD.
        // For Frida, real_pthread_create is populated by the gum_interceptor_replace_fast call.
        real_pthread_create = dlsym(RTLD_NEXT, "pthread_create");
        if (!real_pthread_create) {
            fprintf(stderr, "SCILIB: Critical error: dlsym failed to find real pthread_create.\n");
            // This is a fatal error for LD_PRELOAD if it happens.
            // For Frida, if gum_interceptor_replace_fast failed, this might also be null.
            return -1; // Indicate error
        }
    }

    thread_starter_args_t *wrapper_args = (thread_starter_args_t *)malloc(sizeof(thread_starter_args_t));
    if (!wrapper_args) {
        fprintf(stderr, "SCILIB: Failed to allocate memory for thread wrapper args. Calling original pthread_create.\n");
        return real_pthread_create(thread, attr, start_routine, arg);
    }
    wrapper_args->user_start_routine = start_routine;
    wrapper_args->user_arg = arg;

    return real_pthread_create(thread, attr, thread_starter_wrapper, wrapper_args);
}

void scilib_pthread_wrap_init(void) {
    // For the main thread, assign a stream now.
    // Ensure scilib_nvidia_init (which creates streams) has been called before this.
    assign_stream_to_current_thread();

    // If not using LD_PRELOAD for pthread_create (i.e., using Frida),
    // real_pthread_create will be set by Frida.
    // If using LD_PRELOAD, real_pthread_create will be set on the first call
    // to the wrapper if not already set.
}