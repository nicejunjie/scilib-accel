
#define _GNU_SOURCE 

#include <stdio.h> 
#include <dlfcn.h>
#include <stdlib.h>

#include <complex.h>


#include <time.h>
#include <sys/time.h>

double myt_app=0.0;
double myt_sgemm=0.0;
double myt_dgemm=0.0;
double myt_cgemm=0.0;
double myt_zgemm=0.0;
int myn_sgemm=0;
int myn_dgemm=0;
int myn_cgemm=0;
int myn_zgemm=0;

double scilib_second()
{
        struct timeval tp;
        struct timezone tzp;
        int i;
        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

double scilib_second2()
{
    struct timespec measure;
    clock_gettime(CLOCK_MONOTONIC, &measure);
    return (double)measure.tv_sec + (double)measure.tv_nsec * 1e-9;
}

double scilib_second_() {return scilib_second();}

void sgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, 
                const float *alpha, const float *a, const int *lda, const double *b, const int *ldb, 
                const float *beta, float *c, const int *ldc) 
{
   myt_sgemm-=scilib_second2();
   myn_sgemm++;
#ifdef DEBUG
   fprintf(stderr,"MYPROF sgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, alpha=%.1f, lda=%d, ldb=%d, beta=%.1f, ldc=%d\n",
        *transa, *transb, *m, *n, *k, *alpha, *lda, *ldb, *beta, *ldc);
#endif
   static void (*orig_f)()=NULL;
   if (!orig_f) orig_f = dlsym(RTLD_NEXT, __func__);
   orig_f(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   myt_sgemm+=scilib_second2();
   return;
}

void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, 
                const double *alpha, const double *a, const int *lda, const double *b, const int *ldb, 
                const double *beta, double *c, const int *ldc) 
{
   myt_dgemm-=scilib_second2();
   myn_dgemm++;
#ifdef DEBUG
   fprintf(stderr,"MYPROF dgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, alpha=%.1f, lda=%d, ldb=%d, beta=%.1f, ldc=%d\n",
        *transa, *transb, *m, *n, *k, *alpha, *lda, *ldb, *beta, *ldc);
#endif
   static void (*orig_f)()=NULL;
   if (!orig_f) orig_f = dlsym(RTLD_NEXT, __func__);
   orig_f(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
   myt_dgemm+=scilib_second2();
   return;
}

void cgemm_( const char* transa, const char* transb, const int* m, const int* n, const int *k,
                 const void* alpha, const void* A, const int* lda, const void* B, const int *ldb,
                 const void* beta, void* C, const int* ldc) 
{
   myt_cgemm-=scilib_second2();
   myn_cgemm++;
#ifdef DEBUG
   fprintf(stderr,"MYPROF cgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, alpha=(%.1f, %.1f), lda=%d, ldb=%d, beta=(%.1f, %.1f), ldc=%d\n",
           *transa, *transb, *m, *n, *k, creal(*((float complex*)alpha)), cimag(*((float complex*)alpha)), *lda, *ldb, creal(*((float complex*)beta)), cimag(*((float complex*)beta)), *ldc);
#endif
   static void (*orig_f)()=NULL;
   if (!orig_f) orig_f = dlsym(RTLD_NEXT, __func__);
   orig_f(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
   myt_cgemm+=scilib_second2();
   return;
}

void zgemm_( const char* transa, const char* transb, const int* m, const int* n, const int *k,
                 const void* alpha, const void* A, const int* lda, const void* B, const int *ldb,
                 const void* beta, void* C, const int* ldc) 
{
   myt_zgemm-=scilib_second2();
   myn_zgemm++;
#ifdef DEBUG
   fprintf(stderr,"MYPROF zgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, alpha=(%.1f, %.1f), lda=%d, ldb=%d, beta=(%.1f, %.1f), ldc=%d\n",
           *transa, *transb, *m, *n, *k, creal(*((double complex*)alpha)), cimag(*((double complex*)alpha)), *lda, *ldb, creal(*((double complex*)beta)), cimag(*((double complex*)beta)), *ldc);
#endif
   static void (*orig_f)()=NULL;
   if (!orig_f) orig_f = dlsym(RTLD_NEXT, __func__);
   orig_f(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
   myt_zgemm+=scilib_second2();
   return;
}




int check_MPI() {
    char* pmi_rank = getenv("PMI_RANK");
    //char* pmix_rank = getenv("MPIX_RANK");
    char* mvapich_rank = getenv("MV2_COMM_WORLD_RANK");
    char* ompi_rank = getenv("OMPI_COMM_WORLD_RANK");
    //char* slurm_rank = getenv("SLURM_PROCID");

    if (pmi_rank != NULL  || mvapich_rank != NULL || ompi_rank != NULL )
        return 1;
    else
        return 0;
}

int get_MPI_rank() {
    int rank = -1;
    if (getenv("PMI_RANK") != NULL) {
        rank = atoi(getenv("PMI_RANK"));
    } else if (getenv("MV2_COMM_WORLD_RANK") != NULL) {
        rank = atoi(getenv("MV2_COMM_WORLD_RANK"));
    } else if (getenv("OMPI_COMM_WORLD_RANK") != NULL) {
        rank = atoi(getenv("OMPI_COMM_WORLD_RANK"));
    }
    return rank;
}



void myinit(){
   myt_app -= scilib_second2();
}

void myfini(){
   myt_app += scilib_second2();
   if(myt_sgemm>0.000001) fprintf(stderr,"MYPROF sgemm: count = %d , time = %.6f\n", myn_sgemm, myt_sgemm);
   if(myt_dgemm>0.000001) fprintf(stderr,"MYPROF dgemm: count = %d , time = %.6f\n", myn_dgemm, myt_dgemm);
   if(myt_cgemm>0.000001) fprintf(stderr,"MYPROF cgemm: count = %d , time = %.6f\n", myn_cgemm, myt_cgemm);
   if(myt_zgemm>0.000001) fprintf(stderr,"MYPROF zgemm: count = %d , time = %.6f\n", myn_zgemm, myt_zgemm);
   if (myt_sgemm>0.000001 || myt_dgemm>0.000001 || myt_cgemm>0.000001 || myt_zgemm>0.000001 ) fprintf(stderr,"MYPROF Total: time = %.6f\n",  myt_app);  
}



//  __attribute__((section(".init_array"))) void *__init = mylib_init;
// __attribute__((section(".fini_array"))) void *__fini = myfini;


  void myinit() __attribute__((constructor));
  void myfini() __attribute__((destructor));
