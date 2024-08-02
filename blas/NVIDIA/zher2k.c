#include "myblas.h"

#ifdef DBI
#define _ZHER2K myzher2k
#else 
#define _ZHER2K zher2k_
#endif 

void _ZHER2K(const char *uplo, const char *trans, const int *n, const int *k, const void *alpha, const void* A,
             const int *lda, const void* B, const int *ldb, const void *beta, void* C, const int *ldc) {

    enum findex fi = zher2k; 
    static void (*orig_f)() = NULL; 
    double t0=0.0, t1=0.0;

    DEBUG1(t0 -= mysecond());

    double avgn = cbrt((double)*n * (double)*n * (double)*k);

    int size_type = sizeof(cuDoubleComplex);
    size_t sizeA = (*lda) * ((trans[0] == 'N' || trans[0] == 'n') ? *k : *n);
    size_t sizeB = (*ldb) * ((trans[0] == 'N' || trans[0] == 'n') ? *k : *n);
    size_t sizeC = (*ldc) * (*n);
    sizeA *= size_type;
    sizeB *= size_type;
    sizeC *= size_type;

    double matrix_mem_size_mb = ((double)sizeA + (double)sizeB + (double)sizeC) / 1024.0 / 1024.0;
    double beta_abs = cuCabs(*((cuDoubleComplex*) beta));

    if(avgn < scilib_matrix_offload_size) {
        DEBUG2(fprintf(stderr,"cpu: zher2k args: uplo=%c, trans=%c, n=%d, k=%d, alpha=(%.1e, %.1e), \
          lda=%d, ldb=%d, beta=(%.1e, %.1e), ldc=%d\n",
          *uplo, *trans, *n, *k, creal(*(double complex*)alpha), cimag(*(double complex*)alpha),
          *lda, *ldb, creal(*(double complex*)beta), cimag(*(double complex*)beta), *ldc));

        if (!orig_f) orig_f = farray[fi].fptr;
        DEBUG1(t1 -= mysecond());
        orig_f(uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

        double ts;
        DEBUG1(ts = mysecond());
        DEBUG1(t1 += ts);
        DEBUG1(t0 += ts);

        DEBUG3(fprintf(stderr, "cpu: single zher2k timing(s): total= %10.6f\n", t0 ));

        DEBUG1(farray[fi].t0 += t0);
        DEBUG1(farray[fi].t1 += t1);

        return;
    }

    DEBUG2(fprintf(stderr,"gpu: zher2k args: uplo=%c, trans=%c, n=%d, k=%d, alpha=(%.1e, %.1e), \
      lda=%d, ldb=%d, beta=(%.1e, %.1e), ldc=%d\n",
      *uplo, *trans, *n, *k, creal(*(double complex*)alpha), cimag(*(double complex*)alpha),
      *lda, *ldb, creal(*(double complex*)beta), cimag(*(double complex*)beta), *ldc));

    cublasFillMode_t gpu_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t gpu_trans = (trans[0] == 'N' || trans[0] == 'n') ? CUBLAS_OP_N : CUBLAS_OP_C;

    if(scilib_offload_mode == 1) {
        cuDoubleComplex *d_A, *d_B, *d_C;

        CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeA, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&d_B, sizeB, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&d_C, sizeC, stream));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeA, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeB, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_C, C, sizeC, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaDeviceSynchronize());

        DEBUG1(t1 -= mysecond());
        CUBLAS_CHECK(cublasZher2k(handle, gpu_uplo, gpu_trans, *n, *k, alpha, d_A, *lda, d_B, *ldb, beta, d_C, *ldc));
        CUDA_CHECK(cudaDeviceSynchronize());
        DEBUG1(t1 += mysecond());
        CUDA_CHECK(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFreeAsync(d_A, stream));
        CUDA_CHECK(cudaFreeAsync(d_B, stream));
        CUDA_CHECK(cudaFreeAsync(d_C, stream));
    }
    else {
        int inumaA, inumaB, inumaC;
        if (scilib_offload_mode == 3) {
            inumaA = which_numa(A, sizeA);
            inumaB = which_numa(B, sizeB);
            inumaC = which_numa(C, sizeC);
            DEBUG3(fprintf(stderr,"a,NUMA location of A,B,C: %d %d %d\n", inumaA, inumaB, inumaC));
            if (inumaA == 0) move_numa(A, sizeA, NUMA_HBM);
            if (inumaB == 0) move_numa(B, sizeB, NUMA_HBM);
            if (inumaC == 0) move_numa(C, sizeC, NUMA_HBM);
            DEBUG3(fprintf(stderr,"b,NUMA location of A,B,C: %d %d %d\n", inumaA, inumaB, inumaC));
        }

        DEBUG1(t1 -= mysecond());
        CUBLAS_CHECK(cublasZher2k(handle, gpu_uplo, gpu_trans, *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc));
        CUDA_CHECK(cudaDeviceSynchronize());
        DEBUG3(fprintf(stderr,"c,NUMA location of A,B,C: %d %d %d\n", inumaA, inumaB, inumaC));
        DEBUG1(t1 += mysecond());
    }

    DEBUG1(t0 += mysecond());

    DEBUG3(fprintf(stderr, "gpu: single zher2k timing(s): total= %10.6f, compute= %10.6f, other= %10.6f\n", t0, t1, t0-t1));

    DEBUG1(farray[fi].t0 += t0);
    DEBUG1(farray[fi].t1 += t1);

    return;
}

