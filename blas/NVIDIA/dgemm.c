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
    double t0=0.0, t1=0.0;

    DEBUG1(t0 -= scilib_second());

    double avgn=cbrt(*m)*cbrt(*n)*cbrt(*k);

    int size_type = sizeof(double);
    size_t sizeA = (transa[0] == 'N'||transa[0] == 'n') ? ((*k) * (*lda)) : ((*m) * (*lda));
    size_t sizeB = (transb[0] == 'N'||transb[0] == 'n') ? ((*n) * (*ldb)) : ((*k) * (*ldb));
    size_t sizeC = (*n) * (*ldc);
    sizeA *= size_type;
    sizeB *= size_type;
    sizeC *= size_type;
    double matrix_mem_size_mb = ((double)sizeA+(double)sizeB+(double)sizeC) / 1024.0 / 1024.0;
    double beta_abs = fabs( *beta);
    int ic = (beta_abs > 1.0e-8) ? 2:1;
    double matrix_mem_size_mb_copy = ((double)sizeA+(double)sizeB+(double)sizeC*ic) / 1024.0 / 1024.0;

    if(avgn<scilib_matrix_offload_size)  {
         DEBUG2(fprintf(stderr,"cpu: dgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, alpha=%.1e, lda=%d, ldb=%d, beta=%.ef, ldc=%d\n",
           *transa, *transb, *m, *n, *k, *alpha, *lda, *ldb, *beta, *ldc));

         if (!orig_f) orig_f = scilib_farray[fi].fptr;
         DEBUG1(t1 -= scilib_second());
         orig_f(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
         double ts;
         DEBUG1(ts = scilib_second());
         DEBUG1(t1 += ts);
         DEBUG1(t0 += ts);

         DEBUG3(fprintf(stderr, "cpu: single dgemm timing(s): total= %10.6f\n", t0 ));

         DEBUG1(scilib_farray[fi].t0 += t0);
         DEBUG1(scilib_farray[fi].t1 += t1);

         return;
    }
    DEBUG2(fprintf(stderr,"gpu: dgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, alpha=%.1e, lda=%d, ldb=%d, beta=%.1e, ldc=%d\n",
        *transa, *transb, *m, *n, *k, *alpha, *lda, *ldb, *beta, *ldc));

    cublasOperation_t transA = (transa[0] == 'N' || transa[0] == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transB = (transb[0] == 'N' || transb[0] == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

if(scilib_offload_mode == 1){
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

    DEBUG1(t1 -= scilib_second());
    CUBLAS_CHECK(cublasDgemm(handle, transA, transB, *m, *n, *k, alpha, d_A, *lda, d_B, *ldb, beta, d_C, *ldc));
    CUDA_CHECK(cudaDeviceSynchronize());
    DEBUG1(t1 += scilib_second());
    CUDA_CHECK(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFreeAsync(d_A, stream));
    CUDA_CHECK(cudaFreeAsync(d_B, stream));
    CUDA_CHECK(cudaFreeAsync(d_C, stream));

}
else {
    int inumaA, inumaB, inumaC;
    if (scilib_offload_mode == 3) {
       inumaA=which_numa(A, sizeA);
       inumaB=which_numa(B, sizeB);
       inumaC=which_numa(C, sizeC);
       DEBUG3(fprintf(stderr,"a,NUMA location of A,B,C: %d %d %d\n", inumaA, inumaB, inumaC));
       if ( inumaA == 0 ) move_numa(A, sizeA, NUMA_HBM);
       if ( inumaB == 0 ) move_numa(B, sizeB, NUMA_HBM);
       if ( inumaC == 0 ) move_numa(C, sizeC, NUMA_HBM);
       DEBUG3(fprintf(stderr,"b,NUMA location of A,B,C: %d %d %d\n", inumaA, inumaB, inumaC));
    }

    DEBUG1(t1 -= scilib_second());
    CUBLAS_CHECK(cublasDgemm(handle, transA, transB, *m, *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc));
    CUDA_CHECK(cudaDeviceSynchronize());
       DEBUG3(fprintf(stderr,"c,NUMA location of A,B,C: %d %d %d\n", inumaA, inumaB, inumaC));
    DEBUG1(t1 += scilib_second());
}

    DEBUG1(t0 += scilib_second());

    DEBUG3(fprintf(stderr, "gpu: single dgemm timing(s): total= %10.6f, compute= %10.6f, other= %10.6f\n", t0, t1, t0-t1));

    DEBUG1(scilib_farray[fi].t0 += t0);
    DEBUG1(scilib_farray[fi].t1 += t1);

    return;
}

