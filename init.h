
void init(); 

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
// level-2 BLAS
  sgbmv,
  dgbmv,
  cgbmv,
  zgbmv,
  sgemv,
  dgemv,
  cgemv,
  zgemv,
  sger,
  dger,
  cgerc,
  zgerc,
  cgeru,
  zgeru,
  chbmv,
  zhbmv,
  chemv,
  zhemv,
  cher,
  zher,
  cher2,
  zher2,
  chpmv,
  zhpmv,
  chpr,
  zhpr,
  chpr2,
  zhpr2,
  ssbmv,
  dsbmv,
  sspmv,
  dspmv,
  sspr,
  dspr,
  sspr2,
  dspr2,
  ssymv,
  dsymv,
  ssyr,
  dsyr,
  ssyr2,
  dsyr2,
  stbmv,
  dtbmv,
  ctbmv,
  ztbmv,
  stbsv,
  dtbsv,
  ctbsv,
  ztbsv,
  stpmv,
  dtpmv,
  ctpmv,
  ztpmv,
  stpsv,
  dtpsv,
  ctpsv,
  ztpsv,
  strmv,
  dtrmv,
  ctrmv,
  ztrmv,
  strsv,
  dtrsv,
  ctrsv,
  ztrsv,
};

#define X(f0_val, f1_val) { f0_val, f1_val, NULL, 0.0, 0.0, 0 }
#define INIT_BLAS_L3 \
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
    X("zher2k_", "myzher2k")

#define INIT_BLAS_L2 \
    X("sgbmv_", "mysgbmv"), \
    X("dgbmv_", "mydgbmv"), \
    X("cgbmv_", "mycgbmv"), \
    X("zgbmv_", "myzgbmv"), \
    X("sgemv_", "mysgemv"), \
    X("dgemv_", "mydgemv"), \
    X("cgemv_", "mycgemv"), \
    X("zgemv_", "myzgemv"), \
    X("sger_", "mysger"), \
    X("dger_", "mydger"), \
    X("cgerc_", "mycgerc"), \
    X("zgerc_", "myzgerc"), \
    X("cgeru_", "mycgeru"), \
    X("zgeru_", "myzgeru"), \
    X("chbmv_", "mychbmv"), \
    X("zhbmv_", "myzhbmv"), \
    X("chemv_", "mychemv"), \
    X("zhemv_", "myzhemv"), \
    X("cher_", "mycher"), \
    X("zher_", "myzher"), \
    X("cher2_", "mycher2"), \
    X("zher2_", "myzher2"), \
    X("chpmv_", "mychpmv"), \
    X("zhpmv_", "myzhpmv"), \
    X("chpr_", "mychpr"), \
    X("zhpr_", "myzhpr"), \
    X("chpr2_", "mychpr2"), \
    X("zhpr2_", "myzhpr2"), \
    X("ssbmv_", "myssbmv"), \
    X("dsbmv_", "mydsbmv"), \
    X("sspmv_", "mysspmv"), \
    X("dspmv_", "mydspmv"), \
    X("sspr_", "mysspr"), \
    X("dspr_", "mydspr"), \
    X("sspr2_", "mysspr2"), \
    X("dspr2_", "mydspr2"), \
    X("ssymv_", "myssymv"), \
    X("dsymv_", "mydsymv"), \
    X("ssyr_", "myssyr"), \
    X("dsyr_", "mydsyr"), \
    X("ssyr2_", "myssyr2"), \
    X("dsyr2_", "mydsyr2"), \
    X("stbmv_", "mystbmv"), \
    X("dtbmv_", "mydtbmv"), \
    X("ctbmv_", "myctbmv"), \
    X("ztbmv_", "myztbmv"), \
    X("stbsv_", "mystbsv"), \
    X("dtbsv_", "mydtbsv"), \
    X("ctbsv_", "myctbsv"), \
    X("ztbsv_", "myztbsv"), \
    X("stpmv_", "mystpmv"), \
    X("dtpmv_", "mydtpmv"), \
    X("ctpmv_", "myctpmv"), \
    X("ztpmv_", "myztpmv"), \
    X("stpsv_", "mystpsv"), \
    X("dtpsv_", "mydtpsv"), \
    X("ctpsv_", "myctpsv"), \
    X("ztpsv_", "myztpsv"), \
    X("strmv_", "mystrmv"), \
    X("dtrmv_", "mydtrmv"), \
    X("ctrmv_", "myctrmv"), \
    X("ztrmv_", "myztrmv"), \
    X("strsv_", "mystrsv"), \
    X("dtrsv_", "mydtrsv"), \
    X("ctrsv_", "myctrsv"), \
    X("ztrsv_", "myztrsv")


#define INIT_FARRAY \ 
   INIT_BLAS_L3, \
   INIT_BLAS_L2,

