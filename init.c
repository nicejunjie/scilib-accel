
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>




//environment variables
int scilib_matrix_offload_size;   //SCILIB_MATRIX_OFFLOAD_SIZE
int scilib_debug;                 //SCILIB_DEBUG
int scilib_thpoff;                //SCILIB_THPOFF
int scilib_offload_mode;          //SCILIB_OFFLOAD_MODE

void parse_env_var() {
   
    char* env_str; 

    env_str = getenv("SCILIB_MATRIX_OFFLOAD_SIZE") ;
    scilib_matrix_offload_size = env_str? atoi(env_str) : 500;

    env_str = getenv("SCILIB_DEBUG");
    scilib_debug = env_str ? atoi(env_str) : 0;


    env_str = getenv("SCILIB_THPOFF");
    scilib_thpoff = env_str ? atoi(env_str) : 0;

    env_str = getenv("SCILIB_OFFLOAD_MODE");
    scilib_offload_mode = env_str ? atoi(env_str) : 2;
    
}


void init(){
  parse_env_var();
  return;
}
