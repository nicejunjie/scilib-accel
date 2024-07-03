

//environment variables
extern int scilib_matrix_offload_size ;   //SCILIB_MATRIX_OFFLOAD_SIZE
extern int scilib_debug;                  //SCILIB_DEBUG
extern int scilib_thpoff;                 //SCILIB_THPOFF
extern int scilib_offload_mode;           //SCILIB_OFFLOAD_MODE

extern int skip_flag;

#define NUMA_HBM 1




#define GiB 1024*1024*1024;
#define MiB 1024*1024;
#define KiB 1024;

// Define the DEBUG macro
#define DEBUG1(x) do { if (scilib_debug>=1) { x; } } while (0)
#define DEBUG2(x) do { if (scilib_debug>=2) { x; } } while (0)
#define DEBUG3(x) do { if (scilib_debug>=3) { x; } } while (0)

