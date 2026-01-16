#include "myblas.h"

#ifdef DBI
#define _SSYMM myssymm
#else 
#define _SSYMM ssymm_
#endif 
void _SSYMM(const char *side, const char *uplo, const int *m, const int *n, const float *alpha, const float *A,
            const int *lda, const float *B, const int *ldb, const float *beta, float *C, const int *ldc) {

    enum findex fi = ssymm; 
    static void (*orig_f)() = NULL; 
    double t0=0.0, t1=0.0;

    DEBUG1(t0 -= scilib_second());

    const int *k = (side[0] == 'L' || side[0] == 'l') ? m : n;

    double avgn = cbrt((double)*m * (double)*n * (double)*k);

    int size_type = sizeof(float);
    size_t sizeA = (*k) * (*lda);
    size_t sizeB = (*n) * (*ldb);
    size_t sizeC = (*n) * (*ldc);
    sizeA *= size_type;
    sizeB *= size_type;
    sizeC *= size_type;

    double matrix_mem_size_mb = ((double)sizeA+(double)sizeB+(double)sizeC) / 1024.0 / 1024.0;
    float beta_abs = fabsf( *beta);
    //int ic = (beta_abs > 1.0e-8) ? 2:1;
    //double matrix_mem_size_mb_copy = ((double)sizeA+(double)sizeB+(double)sizeC*ic) / 1024.0 / 1024.0;

    if(avgn<scilib_matrix_offload_size)  {
         DEBUG2(fprintf(stderr,"cpu: ssymm args: side=%c, uplo=%c, m=%d, n=%d, alpha=%.1e, lda=%d, ldb=%d, beta=%.1e, ldc=%d\n",
           *side, *uplo, *m, *n, *alpha, *lda, *ldb, *beta, *ldc));

         if (!orig_f) orig_f = scilib_farray[fi].fptr;
         DEBUG1(t1 -= scilib_second());
         orig_f(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);

         double ts;
         DEBUG1(ts = scilib_second());
         DEBUG1(t1 += ts);
         DEBUG1(t0 += ts);

         DEBUG3(fprintf(stderr, "cpu: single ssymm timing(s): total= %10.6f\n", t0 ));

         DEBUG1(scilib_farray[fi].t0 += t0);
         DEBUG1(scilib_farray[fi].t1 += t1);

         return;
    }
    DEBUG2(fprintf(stderr,"gpu: ssymm args: side=%c, uplo=%c, m=%d, n=%d, alpha=%.1e, lda=%d, ldb=%d, beta=%.1e, ldc=%d\n",
        *side, *uplo, *m, *n, *alpha, *lda, *ldb, *beta, *ldc));

    cublasSideMode_t gpu_side = (side[0] == 'L' || side[0] == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
    cublasFillMode_t gpu_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

if(scilib_offload_mode == 1){
    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeA, scilib_cuda_stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_B, sizeB, scilib_cuda_stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_C, sizeC, scilib_cuda_stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeA, cudaMemcpyHostToDevice, scilib_cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeB, cudaMemcpyHostToDevice, scilib_cuda_stream));
//    if( beta_abs > 1.0e-8 )  bug if gemm on a submatrix
    CUDA_CHECK(cudaMemcpyAsync(d_C, C, sizeC, cudaMemcpyHostToDevice, scilib_cuda_stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    DEBUG1(t1 -= scilib_second());
    CUBLAS_CHECK(cublasSsymm(scilib_cublas_handle, gpu_side, gpu_uplo, *m, *n, alpha, A, *lda, B, *ldb, beta, C, *ldc));
    CUDA_CHECK(cudaDeviceSynchronize());
    DEBUG1(t1 += scilib_second());
    CUDA_CHECK(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFreeAsync(d_A, scilib_cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_B, scilib_cuda_stream));
    CUDA_CHECK(cudaFreeAsync(d_C, scilib_cuda_stream));

}
else {
    int inumaA, inumaB, inumaC;
    if (scilib_offload_mode == 3) {
       inumaA=which_numa(A, sizeA);
       inumaB=which_numa(B, sizeB);
       inumaC=which_numa(C, sizeC);
       DEBUG3(fprintf(stderr,"a,NUMA location of A,B,C: %d %d %d\n", inumaA, inumaB, inumaC));
       if ( inumaA == 0 ) move_numa(A, sizeA, scilib_hbm_numa);
       if ( inumaB == 0 ) move_numa(B, sizeB, scilib_hbm_numa);
       if ( inumaC == 0 ) move_numa(C, sizeC, scilib_hbm_numa);
       DEBUG3(fprintf(stderr,"b,NUMA location of A,B,C: %d %d %d\n", inumaA, inumaB, inumaC));
    }

    DEBUG1(t1 -= scilib_second());
    CUBLAS_CHECK(cublasSsymm(scilib_cublas_handle, gpu_side, gpu_uplo, *m, *n, alpha, A, *lda, B, *ldb, beta, C, *ldc));
    CUDA_CHECK(cudaDeviceSynchronize());
    DEBUG3(fprintf(stderr,"c,NUMA location of A,B,C: %d %d %d\n", inumaA, inumaB, inumaC));
    DEBUG1(t1 += scilib_second());
}

    DEBUG1(t0 += scilib_second());

    DEBUG3(fprintf(stderr, "gpu: single ssymm timing(s): total= %10.6f, compute= %10.6f, other= %10.6f\n", t0, t1, t0-t1));

    DEBUG1(scilib_farray[fi].t0 += t0);
    DEBUG1(scilib_farray[fi].t1 += t1);

    return;
}

