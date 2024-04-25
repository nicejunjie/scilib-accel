#define _GNU_SOURCE

//#include <stdlib.h>
//#include <stdio.h>
#include "mylib.h"
#include <cublas_v2.h>
#include <time.h>
#include <math.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <omp.h>

// disable THP
#include <sys/prctl.h> 

#ifdef INIT_IN_MPI
#include <mpi.h>
#endif


#define GiB 1024*1024*1024;
#define MiB 1024*1024;
#define KiB 1024;

#define CUDA_CHECK(call)                                                  \
do {                                                                      \
    cudaError_t error = call;                                             \
    if (error != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error: %s:%d, ", __FILE__, __LINE__);       \
        fprintf(stderr, "code: %d, reason: %s\n", error,                  \
                cudaGetErrorString(error));                               \
        exit(1);                                                          \
    }                                                                     \
} while (0)

#ifdef PROFILE
#define PROFILE(code) code
#else
#define PROFILE(code)
#endif



int is_MPI=0;
int rank=-1; 
int iprint=0;
unsigned long callcount=0;
 
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



extern double mysecond();

#include <numaif.h>
#include <numa.h>
#include <sys/mman.h>
#include <unistd.h>
#define NUMA_HBM 1
//#define PAGE_SIZE sysconf(_SC_PAGESIZE)


double mtime_total=0.0;
double mtime_comput=0.0;
double mtime_dmove=0.0;
double mvol_dmove=0.0; //in GB

int which_numa(double *var) {
 void * ptr_to_check = var;
 int status[1];
 int ret_code;
 status[0]=-1;
 ret_code=move_pages(0 /*self memory */, 1, &ptr_to_check, NULL, status, 0);
 // this print may cause extra NUMA traffic
 // if(debug) printf("Memory at %p is at numa node %d (retcode %d)\n", ptr_to_check, status[0], ret_code);
 return status[0];
}

void move_numa(double *ptr, size_t size, int target_node) {
// size in Bytes
    //printf("size in move_numa=%d, array size=%d\n",size, size/8);
    double tnuma=mysecond();
    int PAGE_SIZE = getpagesize();
    int rc=0;
    size_t num_pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
    int *status = malloc(num_pages*sizeof(int));
    int *nodes = malloc(num_pages*sizeof(int));
    void **page_addrs = malloc(num_pages * sizeof(void *));

    // Populate the array with page addresses
    #pragma omp parallel for
    for (size_t i = 0; i < num_pages; i++) {
        page_addrs[i] = ptr + (i * PAGE_SIZE / sizeof(double));
        nodes[i]=target_node;
        status[i]=-1;
    }
//    rc=move_pages(0 /*self memory*/, num_pages, page_addrs, nodes, status, 0);
/*
    if(rc!=0) {
        if(rc > 0) fprintf(stderr, "warning: %d pages not moved\n", rc); 
        if(rc < 0) {fprintf(stderr, "error: page migration failed\n"); exit(-1);} 
    }
    free(page_addrs);
*/


/// test 
    #pragma omp parallel
    {
        int thread_rc = 0;
        #pragma omp for
        for (size_t i = 0; i < num_pages; i += omp_get_num_threads()) {
            size_t start = i;
            size_t end = ((i + omp_get_num_threads()) < num_pages) ? (i + omp_get_num_threads()) : num_pages;
            thread_rc = move_pages(0 /*self memory*/, end - start, &page_addrs[start], &nodes[start], &status[start], 0);
            if (thread_rc != 0) {
                #pragma omp critical
                {
                    if (thread_rc > 0) fprintf(stderr, "warning: %d pages not moved\n", thread_rc);
                    if (thread_rc < 0) {
                        fprintf(stderr, "error: page migration failed\n");
                        exit(-1);
                    }
                }
            }
        }
        #pragma omp critical
        {
            rc += thread_rc;
        }
    }
/// test


    tnuma=mysecond()-tnuma;
    //printf("element numa\n");
    //for (int i =0; i<size/8; i++) printf("%d %d\n",i,which_numa(ptr+i));
    printf("move_numa time %15.6f of %lu pages\n", tnuma, num_pages);
    mtime_dmove+=tnuma;
    return;
}


static void (*orig_zgemm)()=NULL; 
cublasStatus_t status;
cublasHandle_t handle;
cudaStream_t stream;






void zgemm_( const char* transa, const char* transb, const int* m, const int* n, const int* k, 
                 const void* alpha, const void* A, const int* lda, const void* B, const int* ldb, 
                 const void* beta, void* C, const int* ldc) {

    mtime_total-=mysecond();
   iprint=0;

   double avgn=cbrt(*m)*cbrt(*n)*cbrt(*k);

   int size_type = sizeof(cuDoubleComplex); //for complex
   size_t sizeA = (transa[0] == 'N'||transa[0] == 'n') ? ((*k) * (*lda)) : ((*m) * (*lda));
   size_t sizeB = (transb[0] == 'N'||transb[0] == 'n') ? ((*n) * (*ldb)) : ((*k) * (*ldb));
   size_t sizeC = (*n) * (*ldc);
   sizeA *= size_type;
   sizeB *= size_type;
   sizeC *= size_type;
   double zgemm_mem_size_mb = ((double)sizeA+(double)sizeB+(double)sizeC) / 1024.0 / 1024.0;
   cuDoubleComplex *beta2=(cuDoubleComplex *)beta;
   double beta_abs = cuCabs( *beta2);
   int ic = (beta_abs > 0.00000001) ? 2:1; 
   mvol_dmove+=((double)sizeA+(double)sizeB+(ic)*(double)sizeC)/1024.0/1024.0/1024.0; 

    if(avgn<500)  {
    //     printf("%s %.1f\n", "zgemm on cpu", avgn);
         mtime_comput-=mysecond();
         orig_zgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); 
         mtime_comput+=mysecond();
         mtime_total+=mysecond();
         return;
    }

   printf("gpu: zgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, lda=%d, ldb=%d, ldc=%d\n",
        *transa, *transb, *m, *n, *k, *lda, *ldb, *ldc);
/*
   // alpla and beta are complex
   printf("gpu: zgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, alpha=%.1f, lda=%d, ldb=%d, beta=%.1f, ldc=%d\n",
        *transa, *transb, *m, *n, *k, *alpha, *lda, *ldb, *beta, *ldc);
*/


    cublasOperation_t transA = (transa[0] == 'N' || transa[0] == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transB = (transb[0] == 'N' || transb[0] == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T; // forgot about H

/*
#ifdef GPUCOPY
    cuDoubleComplex *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeA, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_B, sizeB, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_C, sizeC, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeB, cudaMemcpyHostToDevice, stream));

//    if( beta_abs > 1.0e-8 ) 
         CUDA_CHECK(cudaMemcpyAsync(d_C, C, sizeC, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    status = cublasZgemm(handle, transA, transB, *m, *n, *k, alpha, d_A, *lda, d_B, *ldb, beta, d_C, *ldc);
    CUDA_CHECK(cudaDeviceSynchronize());
   
    CUDA_CHECK(cudaMemcpyAsync(C, d_C, sizeC, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFreeAsync(d_A, stream);
    cudaFreeAsync(d_B, stream);
    cudaFreeAsync(d_C, stream);
    CUDA_CHECK(cudaDeviceSynchronize());
#else  //not GPUCPOY
*/

#ifdef GPUCOPY
    cuDoubleComplex *d_A, *d_B, *d_C;
    //CUDA_CHECK(cudaMallocAsync((cuDoubleComplex **)&d_A, sizeA, stream));
    //CUDA_CHECK(cudaMallocAsync((cuDoubleComplex **)&d_B, sizeB, stream));
    //CUDA_CHECK(cudaMallocAsync((cuDoubleComplex **)&d_C, sizeC, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeA, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_B, sizeB, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_C, sizeC, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeB, cudaMemcpyHostToDevice, stream));
  
    if( beta_abs > 1.0e-8 ) 
        CUDA_CHECK(cudaMemcpyAsync(d_C, C, sizeC, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaDeviceSynchronize());

         mtime_comput-=mysecond();
    status = cublasZgemm(handle, transA, transB, *m, *n, *k, alpha, d_A, *lda, d_B, *ldb, beta, d_C, *ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS error: %d\n", status);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

         mtime_comput+=mysecond();
   
    //CUDA_CHECK(cudaMemcpy(C, d_C, (*m) * (*n) * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(C, d_C, sizeC, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_A, stream));
    CUDA_CHECK(cudaFreeAsync(d_B, stream));
    CUDA_CHECK(cudaFreeAsync(d_C, stream));
    CUDA_CHECK(cudaDeviceSynchronize());
#else  //not GPUCPOY


#ifdef AUTO_NUMA
    int inumaA=which_numa(A);
    int inumaB=which_numa(B);
    int inumaC=which_numa(C);
    //printf("numa node of A=%d B=%d C=%d\n", inumaA, inumaB, inumaC);    
    if ( inumaA == 0 ) move_numa(A, (size_t)sizeA, NUMA_HBM);
    if ( inumaB == 0 ) move_numa(B, (size_t)sizeB, NUMA_HBM);
    if ( inumaC == 0 ) move_numa(C, (size_t)sizeC, NUMA_HBM);
#endif
         mtime_comput-=mysecond();
    status = cublasZgemm(handle, transA, transB, *m, *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Error in cublasZgemm\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
         mtime_comput+=mysecond();
#endif


         mtime_total+=mysecond();
    return;
}

void mylib_init(){

// disable THP for auto-page migration
#ifdef AUTO_NUMA
    prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
#endif

// register functions
    orig_zgemm= dlsym(RTLD_NEXT, "zgemm_");



    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS initialization failed\n");
        return;
    }

#ifdef CUDA_ASYNC
       // Create CUDA stream
    cudaStreamCreate(&stream);
#endif
    return;
}
void mylib_fini(){
    cublasDestroy(handle);
#ifdef CUDA_ASYNC
    cudaStreamDestroy(stream);
#endif
   if(mtime_total>0.000001){
              fprintf(stderr,"zgemm time total= %.6f, data=%.6f, compute=%.6f\n", mtime_total,mtime_dmove,mtime_comput);
#ifdef GPU_COPY
              fprintf(stderr, "data vol (GB): %.6f, copy speed GB/s: %.6f\n", mvol_dmove, mvol_dmove/mtime_dmove);
#endif
   }

    fflush(stderr);
    fflush(stdout);

    return;
}

  __attribute__((section(".init_array"))) void *__init = mylib_init;
  __attribute__((section(".fini_array"))) void *__fini = mylib_fini;
 
