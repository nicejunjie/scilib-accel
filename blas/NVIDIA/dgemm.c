#include "init.h"
#include "myblas.h"

#ifdef DBI
#define _DGEMM mydgemm
#else 
#define _DGEMM dgemm_
#endif 
void _DGEMM( const char* transa, const char* transb, const int* m, const int* n, const int* k, 
                 const double* alpha, const double* A, const int* lda, const double* B, const int* ldb, 
                 const double* beta, double* C, const int* ldc) {

    enum findex fi = dgemm; 
    static void (*orig_f)() = NULL; 

    double avgn=cbrt(*m)*cbrt(*n)*cbrt(*k);

    int size_type = sizeof(double);
    size_t sizeA = (transa[0] == 'N'||transa[0] == 'n') ? ((*k) * (*lda)) : ((*m) * (*lda));
    size_t sizeB = (transb[0] == 'N'||transb[0] == 'n') ? ((*n) * (*ldb)) : ((*k) * (*ldb));
    size_t sizeC = (*n) * (*ldc);
    sizeA *= size_type;
    sizeB *= size_type;
    sizeC *= size_type;
    double dgemm_mem_size_mb = ((double)sizeA+(double)sizeB+(double)sizeC) / 1024.0 / 1024.0;
    double beta_abs = fabs( *beta);
    int ic = (beta_abs > 1.0e-8) ? 2:1;

    if(avgn<500)  {
//         printf("%s %.1f\n", "dgemm on cpu", avgn);
         if (!orig_f) orig_f = farray[fi].fptr;
         orig_f(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
         return;
    }
    printf("gpu: dgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, alpha=%.1f, lda=%d, ldb=%d, beta=%.1f, ldc=%d\n",
        *transa, *transb, *m, *n, *k, *alpha, *lda, *ldb, *beta, *ldc);

    cublasOperation_t transA = (transa[0] == 'N' || transa[0] == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transB = (transb[0] == 'N' || transb[0] == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

#ifdef GPUCOPY
/*
        CUDA_CHECK(cudaHostRegister((void*)A, sizeA, cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister((void*)B, sizeB, cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister((void*)C, sizeC, cudaHostRegisterDefault));
*/

    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeA, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_B, sizeB, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_C, sizeC, stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeB, cudaMemcpyHostToDevice, stream));
//    if( beta_abs > 1.0e-8 )  bug if gemm on a submatrix
    CUDA_CHECK(cudaMemcpyAsync(d_C, C, sizeC, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    status = cublasDgemm(handle, transA, transB, *m, *n, *k, alpha, d_A, *lda, d_B, *ldb, beta, d_C, *ldc);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFreeAsync(d_A, stream));
    CUDA_CHECK(cudaFreeAsync(d_B, stream));
    CUDA_CHECK(cudaFreeAsync(d_C, stream));

#else  //not GPUCPOY

#ifdef AUTO_NUMA
    int inumaA=which_numa(A);
    int inumaB=which_numa(B);
    int inumaC=which_numa(C);
    if ( inumaA == 0 ) move_numa(A, sizeA, NUMA_HBM);
    if ( inumaB == 0 ) move_numa(B, sizeB, NUMA_HBM);
    if ( inumaC == 0 ) move_numa(C, sizeC, NUMA_HBM);
#endif

    CUBLAS_CHECK(cublasDgemm(handle, transA, transB, *m, *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc));
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    return;
}




