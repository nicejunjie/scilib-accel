#define _GNU_SOURCE

#include <dlfcn.h>
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


scilib_freplace scilib_farray[] = {
  INIT_FARRAY
};
int scilib_fsize = sizeof(scilib_farray) / sizeof(scilib_farray[0]);

char *exe_path;
int scilib_skip_flag; 

void scilib_elf_init(){

  get_exe_path(&exe_path);
  scilib_skip_flag = check_string(exe_path);

  scilib_init();

  if (scilib_thpoff == 1) 
      prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);


#ifdef NVIDIA
  scilib_nvidia_init();
#endif

// register functions
  for( int i=0; i< scilib_fsize; i++) {
     scilib_farray[i].fptr= dlsym(RTLD_NEXT, scilib_farray[i].f0);
  }

  return;
}


void scilib_elf_fini(){

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
