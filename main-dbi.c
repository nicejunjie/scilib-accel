#include "frida-gum.h"

#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>


#include "utils.h"
#include "init.h"
#include "global.h"

#ifdef NVIDIA
#include "nvidia.h"
#endif

//to disable THP
#include <sys/prctl.h> 
#include "scilib_pthread_wrap.h"

GumInterceptor * interceptor;
gpointer *hook_address;

scilib_freplace scilib_farray[] = {
  INIT_FARRAY
};
int scilib_fsize = sizeof(scilib_farray) / sizeof(scilib_farray[0]);

char *exe_path;
int scilib_skip_flag; 

void scilib_elf_init(){

  get_exe_path(&exe_path);
  scilib_skip_flag = check_string(exe_path);
  if(scilib_skip_flag) return;

  scilib_init();

//  if (getpagesize() == 65536)  // 64K page, turn off THP 
  if (scilib_thpoff == 1) 
      prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);


#ifdef NVIDIA
  scilib_nvidia_init();
#endif

  scilib_pthread_wrap_init();

  hook_address = malloc(scilib_fsize * sizeof(gpointer));

  gum_init_embedded ();
  interceptor = gum_interceptor_obtain ();
  
  gum_interceptor_begin_transaction (interceptor);
  gpointer orig_pthread_create_addr = gum_find_function("pthread_create");
  gpointer new_pthread_create_addr = (gpointer)scilib_pthread_create_wrapper;

  if (orig_pthread_create_addr && new_pthread_create_addr) {
      gum_interceptor_replace_fast(interceptor,
                                  orig_pthread_create_addr,
                                  new_pthread_create_addr,
                                  (gpointer*)&real_pthread_create); // Frida populates our real_pthread_create
  } else {
      g_printerr("SCILIB: Warning - Failed to find and hook pthread_create.\n");
      // Multi-threading stream assignment might not work as expected.
  }
  for( int i=0; i< scilib_fsize; i++) {
     if( !scilib_offload_func || in_str(scilib_farray[i].f0, scilib_offload_func)) {
         hook_address[i] = gum_find_function(scilib_farray[i].f0);
         gpointer newf = gum_find_function(scilib_farray[i].f1);
         if (hook_address[i] && newf) {
   //        g_print ("%s address = %p\n", scilib_farray[i].f0, hook_address[i]);
   //        g_print ("%s address = %p\n", scilib_farray[i].f1, newf);
             gum_interceptor_replace_fast(interceptor, hook_address[i], newf, 
                   (gpointer*)(&(scilib_farray[i].fptr)));
//           g_print ("ori ptr address %p\n",(scilib_farray[i].fptr));
        }
     }
     else 
        hook_address[i] = NULL;
  }
  gum_interceptor_end_transaction (interceptor);

  return;
}


void scilib_elf_fini(){

  if(scilib_skip_flag) return;

  for( int i=0; i< scilib_fsize; i++) {
    if (hook_address[i]) gum_interceptor_revert(interceptor, hook_address[i]);
  }

  g_object_unref (interceptor);
  gum_deinit_embedded ();

#ifdef NVIDIA
  scilib_nvidia_fini();
#endif

  for( int i=0; i< scilib_fsize; i++) {
     if(scilib_farray[i].t0 > 1e-7)
       fprintf(stderr, "%10s time: total= %15.6f, compute= %15.6f, other= %15.6f\n", scilib_farray[i].f0, scilib_farray[i].t0, scilib_farray[i].t1, scilib_farray[i].t0-scilib_farray[i].t1) ;
  }

  fflush(stderr);
  fflush(stdout);
  
  return;
}

__attribute__((section(".init_array"))) void *__init = scilib_elf_init;
__attribute__((section(".fini_array"))) void *__fini = scilib_elf_fini;
