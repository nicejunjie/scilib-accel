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


freplace farray[] = {
  INIT_FARRAY
};
int fsize = sizeof(farray) / sizeof(farray[0]);

void elf_init(){

  init();

  if (scilib_thpoff == 1) 
      prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);


#ifdef NVIDIA
  nvidia_init();
#endif

// register functions
  for( int i=0; i< fsize; i++) {
     farray[i].fptr= dlsym(RTLD_NEXT, farray[i].f0);
  }

  return;
}


void elf_fini(){

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
