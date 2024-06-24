

//environment variables
extern int env_matrix_offload_size ;   //SCILIB_MATRIX_OFFLOAD_SIZE
extern int env_debug;                  //SCILIB_DEBUG
extern int env_thpoff;                 //SCILIB_THPOFF
extern int env_offload_mode;

extern int skip_flag;

#define NUMA_HBM 1




#define GiB 1024*1024*1024;
#define MiB 1024*1024;
#define KiB 1024;

// Define the DEBUG macro
#define DEBUG1(x) do { if (env_debug>=1) { x; } } while (0)
#define DEBUG2(x) do { if (env_debug>=2) { x; } } while (0)

