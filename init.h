
typedef struct {
    char f0[20];
    char f1[20];
    void *fptr;
} freplace ;

extern freplace farray[] ;

enum findex{
  sgemm,
  dgemm,
  zgemm,
};


#define INIT_FARRAY \
    { "sgemm_", "mysgemm", NULL }, \
    { "dgemm_", "mydgemm", NULL }, \
    { "zgemm_", "myzgemm", NULL }, \
    // Add more elements as needed



