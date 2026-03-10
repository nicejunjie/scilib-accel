
typedef struct {
    char f0[20]; // original func
    char f1[20]; // new func 
    void *fptr;  // ptr to the orignal func
    double t0;   // total time for func
    double t1;   // compute only time for func
} scilib_freplace ;

extern scilib_freplace scilib_farray[] ;

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
  sgemm_internal_,
  dgemm_internal_,
  cgemm_internal_,
  zgemm_internal_,
  ssymm_internal_,
  dsymm_internal_,
  csymm_internal_,
  zsymm_internal_,
  ssyrk_internal_,
  dsyrk_internal_,
  csyrk_internal_,
  zsyrk_internal_,
  ssyr2k_internal_,
  dsyr2k_internal_,
  csyr2k_internal_,
  zsyr2k_internal_,
  strmm_internal_,
  dtrmm_internal_,
  ctrmm_internal_,
  ztrmm_internal_,
  strsm_internal_,
  dtrsm_internal_,
  ctrsm_internal_,
  ztrsm_internal_,
  chemm_internal_,
  zhemm_internal_,
  cherk_internal_,
  zherk_internal_,
  cher2k_internal_,
  zher2k_internal_

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
    X("sgemm_internal_", "mysgemm"), \
    X("dgemm_internal_", "mydgemm"), \
    X("cgemm_internal_", "mycgemm"), \
    X("zgemm_internal_", "myzgemm"), \
    X("ssymm_internal_", "myssymm"), \
    X("dsymm_internal_", "mydsymm"), \
    X("csymm_internal_", "mycsymm"), \
    X("zsymm_internal_", "myzsymm"), \
    X("ssyrk_internal_", "myssyrk"), \
    X("dsyrk_internal_", "mydsyrk"), \
    X("csyrk_internal_", "mycsyrk"), \
    X("zsyrk_internal_", "myzsyrk"), \
    X("ssyr2k_internal_", "myssyr2k"), \
    X("dsyr2k_internal_", "mydsyr2k"), \
    X("csyr2k_internal_", "mycsyr2k"), \
    X("zsyr2k_internal_", "myzsyr2k"), \
    X("strmm_internal_", "mystrmm"), \
    X("dtrmm_internal_", "mydtrmm"), \
    X("ctrmm_internal_", "myctrmm"), \
    X("ztrmm_internal_", "myztrmm"), \
    X("strsm_internal_", "mystrsm"), \
    X("dtrsm_internal_", "mydtrsm"), \
    X("ctrsm_internal_", "myctrsm"), \
    X("ztrsm_internal_", "myztrsm"), \
    X("chemm_internal_", "mychemm"), \
    X("zhemm_internal_", "myzhemm"), \
    X("cherk_internal_", "mycherk"), \
    X("zherk_internal_", "myzherk"), \
    X("cher2k_internal_", "mycher2k"), \
    X("zher2k_internal_", "myzher2k"),

    // Add more elements as needed

