#include "myblas.h"

#ifdef DBI
#define _STRMM mystrmm
#else 
#define _STRMM strmm_
#endif 

void _STRMM(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const float *alpha, const float *A,
            const int *lda, float *B, const int *ldb) {

    enum findex fi = strmm; 
    static void (*orig_f)() = NULL; 
    double t0=0.0, t1=0.0;

    DEBUG1(t0 -= scilib_second());

    const int *k = (side[0] == 'L' || side[0] == 'l') ? m : n;

    double avgn = cbrt((double)*m * (double)*n * (double)*k);

    int size_type = sizeof(float);
    size_t sizeA = (*k) * (*lda);
    size_t sizeB = (*n) * (*ldb);
    sizeA *= size_type;
    sizeB *= size_type;

    double matrix_mem_size_mb = ((double)sizeA+(double)sizeB) / 1024.0 / 1024.0;

    if(avgn<scilib_matrix_offload_size)  {
         DEBUG2(fprintf(stderr,"cpu: strmm args: side=%c, uplo=%c, transa=%c, diag=%c, m=%d, n=%d, alpha=%.1e, lda=%d, ldb=%d\n",
           *side, *uplo, *transa, *diag, *m, *n, *alpha, *lda, *ldb));

         if (!orig_f) orig_f = scilib_farray[fi].fptr;
         DEBUG1(t1 -= scilib_second());
         orig_f(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);

         double ts;
         DEBUG1(ts = scilib_second());
         DEBUG1(t1 += ts);
         DEBUG1(t0 += ts);

         DEBUG3(fprintf(stderr, "cpu: single strmm timing(s): total= %10.6f\n", t0 ));

         DEBUG1(scilib_farray[fi].t0 += t0);
         DEBUG1(scilib_farray[fi].t1 += t1);

         return;
    }
    cudaStream_t current_cuda_stream = scilib_get_current_thread_stream();
    DEBUG2(fprintf(stderr,"gpu: strmm args: side=%c, uplo=%c, transa=%c, diag=%c, m=%d, n=%d, alpha=%.1e, lda=%d, ldb=%d\n",
        *side, *uplo, *transa, *diag, *m, *n, *alpha, *lda, *ldb));

    cublasSideMode_t gpu_side = (side[0] == 'L' || side[0] == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t gpu_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t gpu_transa = (transa[0] == 'N' || transa[0] == 'n') ? CUBLAS_OP_N : 
                                   ((transa[0] == 'T' || transa[0] == 't') ? CUBLAS_OP_T : CUBLAS_OP_C);
    cublasDiagType_t gpu_diag = (diag[0] == 'N' || diag[0] == 'n') ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;

if(scilib_offload_mode == 1){
    float *d_A, *d_B;

    CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeA, current_cuda_stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_B, sizeB, current_cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(current_cuda_stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeA, cudaMemcpyHostToDevice, current_cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeB, cudaMemcpyHostToDevice, current_cuda_stream));
    CUDA_CHECK(cudaStreamSynchronize(current_cuda_stream));

    DEBUG1(t1 -= scilib_second());
    CUBLAS_CHECK(cublasSetStream(scilib_cublas_handle, current_cuda_stream));
    CUBLAS_CHECK(cublasStrmm(scilib_cublas_handle, gpu_side, gpu_uplo, gpu_transa, gpu_diag, *m, *n, alpha, d_A, *lda, d_B, *ldb, d_B, *ldb));
    CUDA_CHECK(cudaStreamSynchronize(current_cuda_stream));
    DEBUG1(t1 += scilib_second());
    CUDA_CHECK(cudaMemcpy(B, d_B, sizeB, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFreeAsync(d_A, current_cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_B, current_cuda_stream));

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

    DEBUG1(t1 -= scilib_second());
    CUBLAS_CHECK(cublasSetStream(scilib_cublas_handle, current_cuda_stream));
    CUBLAS_CHECK(cublasStrmm(scilib_cublas_handle, gpu_side, gpu_uplo, gpu_transa, gpu_diag, *m, *n, alpha, A, *lda, B, *ldb, B, *ldb));
    CUDA_CHECK(cudaStreamSynchronize(current_cuda_stream));
    DEBUG3(fprintf(stderr,"c,NUMA location of A,B: %d %d\n", inumaA, inumaB));
    DEBUG1(t1 += scilib_second());
}

    DEBUG1(t0 += scilib_second());

    DEBUG3(fprintf(stderr, "gpu: single strmm timing(s): total= %10.6f, compute= %10.6f, other= %10.6f\n", t0, t1, t0-t1));

    DEBUG1(scilib_farray[fi].t0 += t0);
    DEBUG1(scilib_farray[fi].t1 += t1);

    return;
}
