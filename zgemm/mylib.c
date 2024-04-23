#define _GNU_SOURCE

//#include <stdlib.h>
//#include <stdio.h>
#include "mylib.h"
#include <cublas_v2.h>
#include <time.h>
#include <math.h>
#include <dlfcn.h>
#include <stdbool.h>

#include <mpi.h>

static void (*orig_zgemm)()=NULL; 
cublasStatus_t status;
cublasHandle_t handle;
cudaStream_t stream;

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

void move_numa(double *ptr, unsigned long size, int target_node) {
// size in Bytes
    //printf("size in move_numa=%d, array size=%d\n",size, size/8);
    double tnuma=mysecond();
    int PAGE_SIZE = getpagesize();
    unsigned long num_pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
    int *status = malloc(num_pages*sizeof(int));
    int *nodes = malloc(num_pages*sizeof(int));
    // Allocate an array to store page addresses
    void **page_addrs = malloc(num_pages * sizeof(void *));
    if (page_addrs == NULL) {
        // Handle allocation failure
        return;
    }

    // Populate the array with page addresses
    for (unsigned long i = 0; i < num_pages; i++) {
        page_addrs[i] = ptr + (i * PAGE_SIZE / sizeof(double));
        nodes[i]=target_node;
        status[i]=-1;
    }

    // Call move_pages once with the array of page addresses
    move_pages(0 /*self memory*/, num_pages, page_addrs, nodes, status, 0);
    //printf("status code\n");
    //for (int i =0; i<num_pages; i++) printf("%d:%d\n",i,status[i]);

    // Free the allocated array
    free(page_addrs);

    tnuma=mysecond()-tnuma;
    //printf("element numa\n");
    //for (int i =0; i<size/8; i++) printf("%d %d\n",i,which_numa(ptr+i));
    printf("move_numa time %15.6f of %lu pages\n", tnuma, num_pages);
    return;
}



void zgemm_( const char* transa, const char* transb, const int* m, const int* n, const int* k, 
                 const void* alpha, const void* A, const int* lda, const void* B, const int* ldb, 
                 const void* beta, void* C, const int* ldc) {

    callcount++;
    //if(callcount>90470) exit(0);
    if ( is_MPI && rank==-1 ) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    iprint=((!is_MPI || rank==0) && (callcount<90470 && callcount>90300));
     iprint=0;

    double avgn=cbrt(*m)*cbrt(*n)*cbrt(*k);
    if(avgn<500)  {
         if(iprint) printf("cpu: zgemm: %s %s %d %d %d  mmem: %d MB\n",transa, transb, *m, *n, *k, ((*m)*(*k)+(*k)*(*n)+(*m)*(*n))/1024/1024*8*2);
         orig_zgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); 

       cuDoubleComplex *d_A, *d_B, *d_C;
       d_A=A;
       d_B=B;
       d_C=C;
       if(iprint){
       printf("zgemm-%lu, A matrix on cpu:\n", callcount);
       for (int i=0;i<5;i++) {
          for (int j=0;j<5;j++) { 
            int index=i*(*n)+j;
            printf("%10.6f %10.6f ", cuCreal(d_A[index]), cuCimag(d_A[index]));
          }
          printf("\n");
       }
       printf("zgemm-%lu, B matrix on cpu:\n", callcount);
       for (int i=0;i<5;i++) {
          for (int j=0;j<5;j++) { 
            int index=i*(*n)+j;
            printf("%10.6f %10.6f ", cuCreal(d_B[index]), cuCimag(d_B[index]));
          }
          printf("\n");
       }
       printf("zgemm-%lu, C matrix on cpu:\n", callcount);
       for (int i=0;i<5;i++) {
          for (int j=0;j<5;j++) { 
            int index=i*(*n)+j;
            printf("%10.6f %10.6f ", cuCreal(d_C[index]), cuCimag(d_C[index]));
          }
          printf("\n");
       }
       printf("\n");
       }
    

          return;
    }
    if(iprint) printf("gpu: zgemm: %s %s %d %d %d  mmem: %d MB\n",transa, transb, *m, *n, *k, ((*m)*(*k)+(*k)*(*n)+(*m)*(*n))/1024/1024*8*2);

    cublasOperation_t transA = (transa[0] == 'N' || transa[0] == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transB = (transb[0] == 'N' || transb[0] == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

#ifdef GPUCOPY
    cuDoubleComplex *d_A, *d_B, *d_C;
    cudaMallocAsync((cuDoubleComplex **)&d_A, (*m) * (*k) * sizeof(cuDoubleComplex), stream);
    cudaMallocAsync((cuDoubleComplex **)&d_B, (*k) * (*n) * sizeof(cuDoubleComplex), stream);
    cudaMallocAsync((cuDoubleComplex **)&d_C, (*m) * (*n) * sizeof(cuDoubleComplex), stream);
    cudaMemcpyAsync(d_A, A, (*m) * (*k) * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, B, (*k) * (*n) * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);
    cuDoubleComplex *beta2=(cuDoubleComplex *)beta;
// to test
/*
    double beta_abs = cuCabs(*beta);
    if( beta_abs > 1.0e-8 ) 
*/
               cudaMemcpyAsync(d_C, C, (*m) * (*n) * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream);
    cudaDeviceSynchronize();

    status = cublasZgemm(handle, transA, transB, *m, *n, *k, alpha, d_A, *lda, d_B, *ldb, beta, d_C, *ldc);
    cudaDeviceSynchronize();
   
    cudaMemcpy(C, d_C, (*m) * (*n) * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaFreeAsync(d_A, stream);
    cudaFreeAsync(d_B, stream);
    cudaFreeAsync(d_C, stream);
#else  //not GPUCPOY

#ifdef AUTO_NUMA
    int inumaA=which_numa(A);
    int inumaB=which_numa(B);
    int inumaC=which_numa(C);
    //printf("numa node of A=%d B=%d C=%d\n", inumaA, inumaB, inumaC);    
    if ( inumaA == 0 ) move_numa(A,(unsigned long)(*m)*(*k)*sizeof(double)*2,NUMA_HBM);
    if ( inumaB == 0 ) move_numa(B,(unsigned long)(*k)*(*n)*sizeof(double)*2,NUMA_HBM);
    if ( inumaC == 0 ) move_numa(C,(unsigned long)(*m)*(*n)*sizeof(double)*2,NUMA_HBM);
#endif
    status = cublasZgemm(handle, transA, transB, *m, *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc);
    cudaDeviceSynchronize();
#endif

    cuDoubleComplex *t_A, *t_B, *t_C;
    t_A=A;
    t_B=B;
    t_C=C;
    if(iprint){   
       printf("zgemm-%lu, A matrix on gpu:\n", callcount);
       for (int i=0;i<5;i++) {
          for (int j=0;j<5;j++) { 
            int index=i*(*n)+j;
            printf("%10.6f %10.6f ", cuCreal(t_A[index]), cuCimag(t_A[index]));
          }
          printf("\n");
       }
       printf("zgemm-%lu, B matrix on gpu:\n", callcount);
       for (int i=0;i<5;i++) {
          for (int j=0;j<5;j++) { 
            int index=i*(*n)+j;
            printf("%10.6f %10.6f ", cuCreal(t_B[index]), cuCimag(t_B[index]));
          }
          printf("\n");
       }
       printf("zgemm-%lu, C matrix on gpu:\n", callcount);
       for (int i=0;i<5;i++) {
          for (int j=0;j<5;j++) { 
            int index=i*(*n)+j;
            printf("%10.6f %10.6f ", cuCreal(t_C[index]), cuCimag(t_C[index]));
          }
          printf("\n");
       }
       printf("\n");
    }
    

    return;
}

void mylib_init(){
    orig_zgemm= dlsym(RTLD_NEXT, "zgemm_");
    status = cublasCreate(&handle);
    cudaStreamCreate(&stream);
    is_MPI=check_MPI();
    return;
}
void mylib_fini(){
    cudaStreamDestroy(stream);
    cublasDestroy(handle);
    return;
}

  __attribute__((section(".init_array"))) void *__init = mylib_init;
  __attribute__((section(".fini_array"))) void *__fini = mylib_fini;
 
