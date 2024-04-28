#include "init.h"
#include "myblas.h"

#ifdef GEMM3M
#define _CUBLASZGEMM cublasZgemm3m
#else 
#define _CUBLASZGEMM cublasZgemm
#endif

#ifdef DBI
#define _ZGEMM myzgemm
#else 
#define _ZGEMM zgemm_
#endif 
void _ZGEMM( const char* transa, const char* transb, const int* m, const int* n, const int* k, 
                 const void* alpha, const void* A, const int* lda, const void* B, const int* ldb, 
                 const void* beta, void* C, const int* ldc) {

    enum findex fi = zgemm; 
    static void (*orig_f)() = NULL; 

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

    if(avgn<500)  {
      //   printf("%s %.1f\n", "zgemm on cpu", avgn);
         if (!orig_f) orig_f = farray[fi].fptr;
         orig_f(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); 
         return;
    }

    printf("gpu: zgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, lda=%d, ldb=%d, ldc=%d\n",
        *transa, *transb, *m, *n, *k, *lda, *ldb, *ldc);
/*
   // alpla and beta are complex
   printf("gpu: zgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, alpha=%.1f, lda=%d, ldb=%d, beta=%.1f, ldc=%d\n",
        *transa, *transb, *m, *n, *k, *alpha, *lda, *ldb, *beta, *ldc);
*/
    cublasOperation_t transA;
    if (transa[0] == 'N' || transa[0] == 'n')  
        transA = CUBLAS_OP_N;
    else if (transa[0] == 'T' || transa[0] == 't')  
        transA = CUBLAS_OP_T;
    else if (transa[0] == 'H' || transa[0] == 'h') 
        transA = CUBLAS_OP_C;
    else {
        printf("invalid transA\n");
        exit(1);
    }
    cublasOperation_t transB;
    if (transb[0] == 'N' || transb[0] == 'n')  
        transB = CUBLAS_OP_N;
    else if (transb[0] == 'T' || transb[0] == 't')  
        transB = CUBLAS_OP_T;
    else if (transb[0] == 'C' || transb[0] == 'c') 
        transB = CUBLAS_OP_C;
    else {
        printf("invalid transB\n");
        exit(1);
    }


#ifdef GPUCOPY
    cuDoubleComplex *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeA, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_B, sizeB, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_C, sizeC, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeB, cudaMemcpyHostToDevice, stream));
    if( beta_abs > 1.0e-8 ) 
        CUDA_CHECK(cudaMemcpyAsync(d_C, C, sizeC, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    status = _CUBLASZGEMM(handle, transA, transB, *m, *n, *k, alpha, d_A, *lda, d_B, *ldb, beta, d_C, *ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS error: %d\n", status);
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpyAsync(C, d_C, sizeC, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaFreeAsync(d_A, stream));
    CUDA_CHECK(cudaFreeAsync(d_B, stream));
    CUDA_CHECK(cudaFreeAsync(d_C, stream));
#else  //not GPUCPOY

#ifdef AUTO_NUMA
    int inumaA=which_numa(A);
    int inumaB=which_numa(B);
    int inumaC=which_numa(C);
    if ( inumaA == 0 ) move_numa(A, (size_t)sizeA, NUMA_HBM);
    if ( inumaB == 0 ) move_numa(B, (size_t)sizeB, NUMA_HBM);
    if ( inumaC == 0 ) move_numa(C, (size_t)sizeC, NUMA_HBM);
#endif

    status = _CUBLASZGEMM(handle, transA, transB, *m, *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Error in cublasZgemm\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    return;
}

