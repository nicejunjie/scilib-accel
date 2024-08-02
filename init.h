
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
  cgemm,
  zgemm,
  ssymm,
  dsymm,
  csymm,
  zsymm,
  ssyrk,
  dsyrk,
  csyrk,
  zsyrk,
  ssyr2k,
  dsyr2k,
  csyr2k,
  zsyr2k,
  strmm,
  dtrmm,
  ctrmm,
  ztrmm,
  strsm,
  dtrsm,
  ctrsm,
  ztrsm,
  chemm,
  zhemm,
  cherk,
  zherk,
  cher2k,
  zher2k,
};

#define X(f0_val, f1_val) { f0_val, f1_val, NULL, 0.0, 0.0 }
#define INIT_FARRAY \
    X("sgemm_", "mysgemm"), \
    X("dgemm_", "mydgemm"), \
    X("cgemm_", "mycgemm"), \
    X("zgemm_", "myzgemm"), \
    X("ssymm_", "myssymm"), \
    X("dsymm_", "mydsymm"), \
    X("csymm_", "mycsymm"), \
    X("zsymm_", "myzsymm"), \
    X("ssyrk_", "myssyrk"), \
    X("dsyrk_", "mydsyrk"), \
    X("csyrk_", "mycsyrk"), \
    X("zsyrk_", "myzsyrk"), \
    X("ssyr2k_", "myssyr2k"), \
    X("dsyr2k_", "mydsyr2k"), \
    X("csyr2k_", "mycsyr2k"), \
    X("zsyr2k_", "myzsyr2k"), \
    X("strmm_", "mystrmm"), \
    X("dtrmm_", "mydtrmm"), \
    X("ctrmm_", "myctrmm"), \
    X("ztrmm_", "myztrmm"), \
    X("strsm_", "mystrsm"), \
    X("dtrsm_", "mydtrsm"), \
    X("ctrsm_", "myctrsm"), \
    X("ztrsm_", "myztrsm"), \
    X("chemm_", "mychemm"), \
    X("zhemm_", "myzhemm"), \
    X("cherk_", "mycherk"), \
    X("zherk_", "myzherk"), \
    X("cher2k_", "mycher2k"), \
    X("zher2k_", "myzher2k"), \
    // Add more elements as needed

