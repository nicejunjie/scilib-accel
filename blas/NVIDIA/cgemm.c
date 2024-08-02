#include "myblas.h"

#ifdef GEMM3M
#define _CUBLASCGEMM cublasCgemm3m
#else 
#define _CUBLASCGEMM cublasCgemm
#endif

#ifdef DBI
#define _CGEMM mycgemm
#else 
#define _CGEMM cgemm_
#endif 
void _CGEMM( const char* transa, const char* transb, const int* m, const int* n, const int* k, 
                 const void* alpha, const void* A, const int* lda, const void* B, const int* ldb, 
                 const void* beta, void* C, const int* ldc) {

    enum findex fi = cgemm; 
    static void (*orig_f)() = NULL; 
    double t0=0.0, t1=0.0;

    DEBUG1(t0 -= mysecond());

    double avgn=cbrt(*m)*cbrt(*n)*cbrt(*k);

    int size_type = sizeof(cuFloatComplex);
    size_t sizeA = (transa[0] == 'N'||transa[0] == 'n') ? ((*k) * (*lda)) : ((*m) * (*lda));
    size_t sizeB = (transb[0] == 'N'||transb[0] == 'n') ? ((*n) * (*ldb)) : ((*k) * (*ldb));
    size_t sizeC = (*n) * (*ldc);
    sizeA *= size_type;
    sizeB *= size_type;
    sizeC *= size_type;
    double cgemm_mem_size_mb = ((double)sizeA+(double)sizeB+(double)sizeC) / 1024.0 / 1024.0;
    cuFloatComplex *beta2=(cuFloatComplex *)beta;
    float beta_abs = cuCabsf( *beta2);
    int ic = (beta_abs > 0.00000001) ? 2:1; 

    if(avgn<scilib_matrix_offload_size)  {
    DEBUG2(fprintf(stderr, "cpu: cgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, alpha=(%.1e, %.1e), \  
       lda=%d, ldb=%d, beta=(%.1e, %.1e),ldc=%d\n",
       *transa, *transb, *m, *n, *k, crealf(*(float complex*)alpha), cimagf(*(float complex*)alpha), 
       *lda, *ldb, crealf(*(float complex*)beta), cimagf(*(float complex*)beta), *ldc));

         if (!orig_f) orig_f = farray[fi].fptr;
         DEBUG1(t1 -= mysecond());
         orig_f(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
         double ts;
         DEBUG1(ts = mysecond());
         DEBUG1(t1 += ts);
         DEBUG1(t0 += ts);

         DEBUG3(fprintf(stderr, "cpu: single cgemm timing(s): total= %10.6f\n", t0 ));

         DEBUG1(farray[fi].t0 += t0);
         DEBUG1(farray[fi].t1 += t1);

         return;
    }

    DEBUG2(fprintf(stderr, "gpu: cgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, alpha=(%.1e, %.1e), \  
       lda=%d, ldb=%d, beta=(%.1e, %.1e),ldc=%d\n",
       *transa, *transb, *m, *n, *k, crealf(*(float complex*)alpha), cimagf(*(float complex*)alpha), 
       *lda, *ldb, crealf(*(float complex*)beta), cimagf(*(float complex*)beta), *ldc));
/*
   // alpla and beta are complex
   printf("gpu: cgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, alpha=%.1f, lda=%d, ldb=%d, beta=%.1f, ldc=%d\n",
        *transa, *transb, *m, *n, *k, *alpha, *lda, *ldb, *beta, *ldc);
*/
    cublasOperation_t transA;
    if (transa[0] == 'N' || transa[0] == 'n')  
        transA = CUBLAS_OP_N;
    else if (transa[0] == 'T' || transa[0] == 't')  
        transA = CUBLAS_OP_T;
    else if (transa[0] == 'C' || transa[0] == 'c') 
        transA = CUBLAS_OP_C;
    else {
        printf("Invalid transA value: %c\n", transa[0]);
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
        printf("Invalid transB value: %c\n", transb[0]);
        exit(1);
    }


if (scilib_offload_mode==1) {
    cuFloatComplex *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeA, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_B, sizeB, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_C, sizeC, stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeB, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C, C, sizeC, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    DEBUG1(t1 -= mysecond());
    CUBLAS_CHECK(_CUBLASCGEMM(handle, transA, transB, *m, *n, *k, alpha, d_A, *lda, d_B, *ldb, beta, d_C, *ldc));
    CUDA_CHECK(cudaDeviceSynchronize());
    DEBUG1(t1 += mysecond());
    CUDA_CHECK(cudaMemcpyAsync(C, d_C, sizeC, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFreeAsync(d_A, stream));
    CUDA_CHECK(cudaFreeAsync(d_B, stream));
    CUDA_CHECK(cudaFreeAsync(d_C, stream));

} 
else {
    int inumaA, inumaB, inumaC;
    if(scilib_offload_mode == 3){
       inumaA=which_numa(A, sizeA);
       inumaB=which_numa(B, sizeB);
       inumaC=which_numa(C, sizeC);
       DEBUG3(fprintf(stderr,"a,NUMA location of A,B,C: %d %d %d\n", inumaA, inumaB, inumaC));
       if ( inumaA == 0 ) move_numa(A, (size_t)sizeA, NUMA_HBM);
       if ( inumaB == 0 ) move_numa(B, (size_t)sizeB, NUMA_HBM);
       if ( inumaC == 0 ) move_numa(C, (size_t)sizeC, NUMA_HBM);
       DEBUG3(fprintf(stderr,"b,NUMA location of A,B,C: %d %d %d\n", inumaA, inumaB, inumaC));
    }

    DEBUG1(t1 -= mysecond());
    CUBLAS_CHECK(_CUBLASCGEMM(handle, transA, transB, *m, *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc));
    CUDA_CHECK(cudaDeviceSynchronize());
    DEBUG1(t1 += mysecond());
}

    DEBUG1(t0 += mysecond());

    DEBUG3(fprintf(stderr, "gpu: single cgemm timing(s): total= %10.6f, compute= %10.6f, other= %10.6f\n", t0, t1, t0-t1));

    DEBUG1(farray[fi].t0 += t0);
    DEBUG1(farray[fi].t1 += t1);

    return;
}

