/*
 * Compile with:
 *
 * gcc -ffunction-sections -fdata-sections frida-gum-example.c -o frida-gum-example -L. -lfrida-gum -ldl -lrt -lresolv -lm -pthread -static-libgcc -Wl,-z,noexecstack,--gc-sections
 *
 * Visit https://frida.re to learn more about Frida.
 */

#include "frida-gum.h"

#include <fcntl.h>
#include <unistd.h>

#include "utils.h"
#include "init.h"



//to disable THP
#include <sys/prctl.h> 


GumInterceptor * interceptor;
gpointer *hook_address;

  int count = 0; 
  freplace farray[] = {
    INIT_FARRAY
  };
  int fsize = sizeof(farray) / sizeof(farray[0]);

cublasStatus_t status;
cublasHandle_t handle;
#ifdef GPUCOPY
cudaStream_t stream;
#endif



/*
void (*original_dgemm)(const char *transa, const char *transb, const int *m, const int *n, const int *k, 
                const double *alpha, const double *a, const int *lda, const double *b, const int *ldb, 
                const double *beta, double *c, const int *ldc); 
void mydgemm(const char *transa, const char *transb, const int *m, const int *n, const int *k, 
                const double *alpha, const double *a, const int *lda, const double *b, const int *ldb, 
                const double *beta, double *c, const int *ldc) 
{
   double t;

   //printf("in my dgemm\n");
   count++;
   original_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   return;
}
*/

void my_init(){
  fprintf(stderr,"SCILIB-accel DBI");
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

  hook_address = malloc(fsize * sizeof(gpointer));

  gum_init_embedded ();
  interceptor = gum_interceptor_obtain ();
  
  gum_interceptor_begin_transaction (interceptor);
  for( int i=0; i< fsize; i++) {
     hook_address[i] = gum_find_function(farray[i].f0);
     gpointer newf = gum_find_function(farray[i].f1);
     if (hook_address[i] && newf) {
         g_print ("%s address = %p\n", farray[i].f0, hook_address[i]);
         g_print ("%s address = %p\n", farray[i].f1, newf);
         gum_interceptor_replace_fast(interceptor, hook_address[i], newf, 
               (gpointer*)(&(farray[i].fptr)));
         g_print ("ori ptr address %p\n",(farray[i].fptr));
//       gum_interceptor_replace(interceptor, hook_address, &mydgemm, 
//             NULL, (gpointer*)(&original_dgemm));
     }
  }
  gum_interceptor_end_transaction (interceptor);
}


void my_fini(){
  for( int i=0; i< fsize; i++) {
    if (hook_address[i]) gum_interceptor_revert(interceptor, hook_address[i]);
  }

  g_print ("[*] mydgemm has %u calls\n", count);

  g_object_unref (interceptor);
  gum_deinit_embedded ();

/*  CUBLAS  */
  cublasDestroy(handle);
#ifdef GPUCOPY
  cudaStreamDestroy(stream);
#endif

  fflush(stderr);
  fflush(stdout);
  
  return;
}

__attribute__((section(".init_array"))) void *__init = my_init;
__attribute__((section(".fini_array"))) void *__fini = my_fini;
