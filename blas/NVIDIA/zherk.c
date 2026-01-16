#include "myblas.h"

#ifdef DBI
#define _ZHERK myzherk
#else 
#define _ZHERK zherk_
#endif 

void _ZHERK(const char *uplo, const char *trans, const int *n, const int *k, const void *alpha, const void* A,
            const int *lda, const void *beta, void* C, const int *ldc) {

    enum findex fi = zherk; 
    static void (*orig_f)() = NULL; 
    double t0=0.0, t1=0.0;

    DEBUG1(t0 -= scilib_second());

    double avgn = cbrt((double)*n * (double)*n * (double)*k);

    int size_type = sizeof(cuDoubleComplex);
    size_t sizeA = (*lda) * ((trans[0] == 'N' || trans[0] == 'n') ? *k : *n);
    size_t sizeC = (*n) * (*ldc);
    sizeA *= size_type;
    sizeC *= size_type;

    double matrix_mem_size_mb = ((double)sizeA+(double)sizeC) / 1024.0 / 1024.0;
    double beta_abs = cuCabs(*((cuDoubleComplex*) beta));

    if(avgn < scilib_matrix_offload_size) {
        DEBUG2(fprintf(stderr,"cpu: zherk args: uplo=%c, trans=%c, n=%d, k=%d, alpha=(%.1e, %.1e), \
          lda=%d, beta=(%.1e, %.1e), ldc=%d\n",
          *uplo, *trans, *n, *k, creal(*(double complex*)alpha), cimag(*(double complex*)alpha),
          *lda, creal(*(double complex*)beta), cimag(*(double complex*)beta), *ldc));

        if (!orig_f) orig_f = scilib_farray[fi].fptr;
        DEBUG1(t1 -= scilib_second());
        orig_f(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);

        double ts;
        DEBUG1(ts = scilib_second());
        DEBUG1(t1 += ts);
        DEBUG1(t0 += ts);

        DEBUG3(fprintf(stderr, "cpu: single zherk timing(s): total= %10.6f\n", t0 ));

        DEBUG1(scilib_farray[fi].t0 += t0);
        DEBUG1(scilib_farray[fi].t1 += t1);

        return;
    }

    DEBUG2(fprintf(stderr,"gpu: zherk args: uplo=%c, trans=%c, n=%d, k=%d, alpha=(%.1e, %.1e), \
      lda=%d, beta=(%.1e, %.1e), ldc=%d\n",
      *uplo, *trans, *n, *k, creal(*(double complex*)alpha), cimag(*(double complex*)alpha),
      *lda, creal(*(double complex*)beta), cimag(*(double complex*)beta), *ldc));

    cublasFillMode_t gpu_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t gpu_trans = (trans[0] == 'N' || trans[0] == 'n') ? CUBLAS_OP_N : CUBLAS_OP_C;

    if(scilib_offload_mode == 1) {
        cuDoubleComplex *d_A, *d_C;

        CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeA, scilib_cuda_stream));
        CUDA_CHECK(cudaMallocAsync((void **)&d_C, sizeC, scilib_cuda_stream));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeA, cudaMemcpyHostToDevice, scilib_cuda_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_C, C, sizeC, cudaMemcpyHostToDevice, scilib_cuda_stream));
        CUDA_CHECK(cudaDeviceSynchronize());

        DEBUG1(t1 -= scilib_second());
        CUBLAS_CHECK(cublasZherk(scilib_cublas_handle, gpu_uplo, gpu_trans, *n, *k, alpha, d_A, *lda, beta, d_C, *ldc));
        CUDA_CHECK(cudaDeviceSynchronize());
        DEBUG1(t1 += scilib_second());
        CUDA_CHECK(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFreeAsync(d_A, scilib_cuda_stream));
        CUDA_CHECK(cudaFreeAsync(d_C, scilib_cuda_stream));
    }
    else {
        int inumaA, inumaC;
        if (scilib_offload_mode == 3) {
            inumaA = which_numa(A, sizeA);
            inumaC = which_numa(C, sizeC);
            DEBUG3(fprintf(stderr,"a,NUMA location of A,C: %d %d\n", inumaA, inumaC));
            if (inumaA == 0) move_numa(A, sizeA, scilib_hbm_numa);
            if (inumaC == 0) move_numa(C, sizeC, scilib_hbm_numa);
            DEBUG3(fprintf(stderr,"b,NUMA location of A,C: %d %d\n", inumaA, inumaC));
        }

        DEBUG1(t1 -= scilib_second());
        CUBLAS_CHECK(cublasZherk(scilib_cublas_handle, gpu_uplo, gpu_trans, *n, *k, alpha, A, *lda, beta, C, *ldc));
        CUDA_CHECK(cudaDeviceSynchronize());
        DEBUG3(fprintf(stderr,"c,NUMA location of A,C: %d %d\n", inumaA, inumaC));
        DEBUG1(t1 += scilib_second());
    }

    DEBUG1(t0 += scilib_second());

    DEBUG3(fprintf(stderr, "gpu: single zherk timing(s): total= %10.6f, compute= %10.6f, other= %10.6f\n", t0, t1, t0-t1));

    DEBUG1(scilib_farray[fi].t0 += t0);
    DEBUG1(scilib_farray[fi].t1 += t1);

    return;
}
