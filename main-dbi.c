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

#ifdef NVIDIA
#include "nvidia.h"
#endif

//to disable THP
#include <sys/prctl.h> 


GumInterceptor * interceptor;
gpointer *hook_address;

  freplace farray[] = {
    INIT_FARRAY
  };
  int fsize = sizeof(farray) / sizeof(farray[0]);


void my_init(){
// fprintf(stderr,"SCILIB-accel DBI");
// disable THP for auto-page migration
#ifdef AUTO_NUMA
//    prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
#endif

#ifdef NVIDIA
  nvidia_init();
#endif

  hook_address = malloc(fsize * sizeof(gpointer));

  gum_init_embedded ();
  interceptor = gum_interceptor_obtain ();
  
  gum_interceptor_begin_transaction (interceptor);
  for( int i=0; i< fsize; i++) {
     hook_address[i] = gum_find_function(farray[i].f0);
     gpointer newf = gum_find_function(farray[i].f1);
     if (hook_address[i] && newf) {
   //    g_print ("%s address = %p\n", farray[i].f0, hook_address[i]);
   //    g_print ("%s address = %p\n", farray[i].f1, newf);
         gum_interceptor_replace_fast(interceptor, hook_address[i], newf, 
               (gpointer*)(&(farray[i].fptr)));
//       g_print ("ori ptr address %p\n",(farray[i].fptr));
     }
  }
  gum_interceptor_end_transaction (interceptor);
}


void my_fini(){
  for( int i=0; i< fsize; i++) {
    if (hook_address[i]) gum_interceptor_revert(interceptor, hook_address[i]);
  }

  g_object_unref (interceptor);
  gum_deinit_embedded ();

#ifdef NVIDIA
  nvidia_fini();
#endif

  for( int i=0; i< fsize; i++) {
     if(farray[i].t0 > 1e-7)
       fprintf(stderr, "%10s time: total= %15.6f, compute= %15.6f, other= %15.6f\n", farray[i].f0, farray[i].t0, farray[i].t1, farray[i].t0-farray[i].t1) ;
  }

  fflush(stderr);
  fflush(stdout);
  
  return;
}

__attribute__((section(".init_array"))) void *__init = my_init;
__attribute__((section(".fini_array"))) void *__fini = my_fini;
