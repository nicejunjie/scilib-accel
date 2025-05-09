
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>




//environment variables
int scilib_matrix_offload_size;        //SCILIB_MATRIX_OFFLOAD_SIZE
int scilib_debug;                      //SCILIB_DEBUG
int scilib_thpoff;                     //SCILIB_THPOFF
int scilib_offload_mode;               //SCILIB_OFFLOAD_MODE
char **scilib_offload_func;            //SCILIB_OFFLOAD_FUNC, only these comma separated funcs are intercepted in DBI.
int scilib_num_cuda_streams;           //SCILIB_NUM_STREAMS


void scilib_parse_env_var() {
   
    char* env_str; 

    env_str = getenv("SCILIB_MATRIX_OFFLOAD_SIZE") ;
    scilib_matrix_offload_size = env_str? atoi(env_str) : 500;

    env_str = getenv("SCILIB_DEBUG");
    scilib_debug = env_str ? atoi(env_str) : 0;


    env_str = getenv("SCILIB_THPOFF");
    scilib_thpoff = env_str ? atoi(env_str) : 0;

    env_str = getenv("SCILIB_OFFLOAD_MODE");
    scilib_offload_mode = env_str ? atoi(env_str) : 3;

    env_str = getenv("SCILIB_OFFLOAD_FUNC");
    scilib_offload_func = env_str ? str_split(env_str,',') : NULL; 

    env_str = getenv("SCILIB_NUM_STREAMS") ;
    scilib_num_cuda_streams = env_str? atoi(env_str) : 1;
     
}

/*
       if(scilib_debug >2) {
            printf("scilib_offload_func values:\n");
            for (int i = 0; scilib_offload_func[i] != NULL; i++) {
               printf("%d: %s\n", i, scilib_offload_func[i]);
            }
        }
*/



void scilib_init(){
  scilib_parse_env_var();
  return;
}
