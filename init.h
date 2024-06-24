
typedef struct {
    char f0[20]; // original func
    char f1[20]; // new func 
    void *fptr;  // ptr to the orignal func
    double t0;   // total time for func
    double t1;   // compute only time for func
} freplace ;

extern freplace farray[] ;

enum findex{
  sgemm,
  dgemm,
  zgemm,
};

#define X(f0_val, f1_val) { f0_val, f1_val, NULL, 0.0, 0.0 }
#define INIT_FARRAY \
    X("sgemm_", "mysgemm"), \
    X("dgemm_", "mydgemm"), \
    X("zgemm_", "myzgemm"), \
    // Add more elements as needed

