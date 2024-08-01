#include "myblas.h"

#ifdef DBI
#define _CTRMM myctrmm
#else 
#define _CTRMM ctrmm_
#endif 

void _CTRMM(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const void *alpha, const void *A,
            const int *lda, void *B, const int *ldb) {

    enum findex fi = ctrmm; 
    static void (*orig_f)() = NULL; 
    double t0=0.0, t1=0.0;

    DEBUG1(t0 -= mysecond());

    const int *k = (side[0] == 'L' || side[0] == 'l') ? m : n;

    double avgn = cbrt((double)*m * (double)*n * (double)*k);

    int size_type = sizeof(cuFloatComplex); //for complex
    size_t sizeA = (*k) * (*lda);
    size_t sizeB = (*n) * (*ldb);
    sizeA *= size_type;
    sizeB *= size_type;

    double matrix_mem_size_mb = ((double)sizeA+(double)sizeB) / 1024.0 / 1024.0;

    if(avgn<scilib_matrix_offload_size)  {
         DEBUG2(fprintf(stderr,"cpu: ctrmm args: side=%c, uplo=%c, transa=%c, diag=%c, m=%d, n=%d, alpha=(%.1e, %.1e), lda=%d, ldb=%d\n",
           *side, *uplo, *transa, *diag, *m, *n, crealf(*(float complex*)alpha), cimagf(*(float complex*)alpha), *lda, *ldb));

         if (!orig_f) orig_f = farray[fi].fptr;
         DEBUG1(t1 -= mysecond());
         orig_f(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);

         double ts;
         DEBUG1(ts = mysecond());
         DEBUG1(t1 += ts);
         DEBUG1(t0 += ts);

         DEBUG3(fprintf(stderr, "cpu: single ctrmm timing(s): total= %10.6f\n", t0 ));

         DEBUG1(farray[fi].t0 += t0);
         DEBUG1(farray[fi].t1 += t1);

         return;
    }
    DEBUG2(fprintf(stderr,"gpu: ctrmm args: side=%c, uplo=%c, transa=%c, diag=%c, m=%d, n=%d, alpha=(%.1e, %.1e), lda=%d, ldb=%d\n",
        *side, *uplo, *transa, *diag, *m, *n, crealf(*(float complex*)alpha), cimagf(*(float complex*)alpha), *lda, *ldb));

    cublasSideMode_t gpu_side = (side[0] == 'L' || side[0] == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t gpu_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t gpu_transa = (transa[0] == 'N' || transa[0] == 'n') ? CUBLAS_OP_N : 
                                   ((transa[0] == 'T' || transa[0] == 't') ? CUBLAS_OP_T : CUBLAS_OP_C);
    cublasDiagType_t gpu_diag = (diag[0] == 'N' || diag[0] == 'n') ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;

if(scilib_offload_mode == 1){
    cuFloatComplex *d_A, *d_B;

    CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeA, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_B, sizeB, stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeB, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    DEBUG1(t1 -= mysecond());
    CUBLAS_CHECK(cublasCtrmm(handle, gpu_side, gpu_uplo, gpu_transa, gpu_diag, *m, *n, alpha, d_A, *lda, d_B, *ldb, d_B, *ldb));
    CUDA_CHECK(cudaDeviceSynchronize());
    DEBUG1(t1 += mysecond());
    CUDA_CHECK(cudaMemcpy(B, d_B, sizeB, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFreeAsync(d_A, stream));
    CUDA_CHECK(cudaFreeAsync(d_B, stream));

}
else {
    int inumaA, inumaB;
    if (scilib_offload_mode == 3) {
       inumaA=which_numa(A, sizeA);
       inumaB=which_numa(B, sizeB);
       DEBUG3(fprintf(stderr,"a,NUMA location of A,B: %d %d\n", inumaA, inumaB));
       if ( inumaA == 0 ) move_numa(A, sizeA, NUMA_HBM);
       if ( inumaB == 0 ) move_numa(B, sizeB, NUMA_HBM);
       DEBUG3(fprintf(stderr,"b,NUMA location of A,B: %d %d\n", inumaA, inumaB));
    }

    DEBUG1(t1 -= mysecond());
    CUBLAS_CHECK(cublasCtrmm(handle, gpu_side, gpu_uplo, gpu_transa, gpu_diag, *m, *n, alpha, A, *lda, B, *ldb, B, *ldb));
    CUDA_CHECK(cudaDeviceSynchronize());
    DEBUG3(fprintf(stderr,"c,NUMA location of A,B: %d %d\n", inumaA, inumaB));
    DEBUG1(t1 += mysecond());
}

    DEBUG1(t0 += mysecond());

    DEBUG3(fprintf(stderr, "gpu: single ctrmm timing(s): total= %10.6f, compute= %10.6f, other= %10.6f\n", t0, t1, t0-t1));

    DEBUG1(farray[fi].t0 += t0);
    DEBUG1(farray[fi].t1 += t1);

    return;
}