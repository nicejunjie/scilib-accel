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


GumInterceptor * interceptor;
gpointer *hook_address;

freplace farray[] = {
  INIT_FARRAY
};
int fsize = sizeof(farray) / sizeof(farray[0]);

char *exe_path;
int skip_flag; 

void elf_init(){

  get_exe_path(&exe_path);
  skip_flag = check_string(exe_path);
  if(skip_flag) return;

  init();

//  if (getpagesize() == 65536)  // 64K page, turn off THP 
  if (scilib_thpoff == 1) 
      prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
  

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

  return;
}


void elf_fini(){

  if(skip_flag) return;

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

__attribute__((section(".init_array"))) void *__init = elf_init;
__attribute__((section(".fini_array"))) void *__fini = elf_fini;
