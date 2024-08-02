#include "myblas.h"

#ifdef DBI
#define _CSYRK mycsyrk
#else 
#define _CSYRK csyrk_
#endif 

void _CSYRK(const char *uplo, const char *trans, const int *n, const int *k, const float *alpha, const void* A,
            const int *lda, const float *beta, void* C, const int *ldc) {

    enum findex fi = csyrk; 
    static void (*orig_f)() = NULL; 
    double t0=0.0, t1=0.0;

    DEBUG1(t0 -= mysecond());

    double avgn = cbrt((double)*n * (double)*n * (double)*k);

    int size_type = sizeof(cuFloatComplex);
    size_t sizeA = (*lda) * ((trans[0] == 'N' || trans[0] == 'n') ? *k : *n);
    size_t sizeC = (*n) * (*ldc);
    sizeA *= size_type;
    sizeC *= size_type;

    double matrix_mem_size_mb = ((double)sizeA+(double)sizeC) / 1024.0 / 1024.0;
    float beta_abs = cuCabsf(*((cuFloatComplex*) beta));

    if(avgn < scilib_matrix_offload_size) {
        DEBUG2(fprintf(stderr,"cpu: csyrk args: uplo=%c, trans=%c, n=%d, k=%d, alpha=(%.1e, %.1e), \
          lda=%d, beta=(%.1e, %.1e), ldc=%d\n",
          *uplo, *trans, *n, *k, crealf(*(float complex*)alpha), cimagf(*(float complex*)alpha),
          *lda, crealf(*(float complex*)beta), cimagf(*(float complex*)beta), *ldc));

        if (!orig_f) orig_f = farray[fi].fptr;
        DEBUG1(t1 -= mysecond());
        orig_f(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);

        double ts;
        DEBUG1(ts = mysecond());
        DEBUG1(t1 += ts);
        DEBUG1(t0 += ts);

        DEBUG3(fprintf(stderr, "cpu: single csyrk timing(s): total= %10.6f\n", t0 ));

        DEBUG1(farray[fi].t0 += t0);
        DEBUG1(farray[fi].t1 += t1);

        return;
    }

    DEBUG2(fprintf(stderr,"gpu: csyrk args: uplo=%c, trans=%c, n=%d, k=%d, alpha=(%.1e, %.1e), \
      lda=%d, beta=(%.1e, %.1e), ldc=%d\n",
      *uplo, *trans, *n, *k, crealf(*(float complex*)alpha), cimagf(*(float complex*)alpha),
      *lda, crealf(*(float complex*)beta), cimagf(*(float complex*)beta), *ldc));

    cublasFillMode_t gpu_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t gpu_trans = (trans[0] == 'N' || trans[0] == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

    if(scilib_offload_mode == 1) {
        cuFloatComplex *d_A, *d_C;

        CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeA, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&d_C, sizeC, stream));
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeA, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_C, C, sizeC, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaDeviceSynchronize());

        DEBUG1(t1 -= mysecond());
        CUBLAS_CHECK(cublasCsyrk(handle, gpu_uplo, gpu_trans, *n, *k, alpha, d_A, *lda, beta, d_C, *ldc));
        CUDA_CHECK(cudaDeviceSynchronize());
        DEBUG1(t1 += mysecond());
        CUDA_CHECK(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFreeAsync(d_A, stream));
        CUDA_CHECK(cudaFreeAsync(d_C, stream));
    }
    else {
        int inumaA, inumaC;
        if (scilib_offload_mode == 3) {
            inumaA = which_numa(A, sizeA);
            inumaC = which_numa(C, sizeC);
            DEBUG3(fprintf(stderr,"a,NUMA location of A,C: %d %d\n", inumaA, inumaC));
            if (inumaA == 0) move_numa(A, sizeA, NUMA_HBM);
            if (inumaC == 0) move_numa(C, sizeC, NUMA_HBM);
            DEBUG3(fprintf(stderr,"b,NUMA location of A,C: %d %d\n", inumaA, inumaC));
        }

        DEBUG1(t1 -= mysecond());
        CUBLAS_CHECK(cublasCsyrk(handle, gpu_uplo, gpu_trans, *n, *k, alpha, A, *lda, beta, C, *ldc));
        CUDA_CHECK(cudaDeviceSynchronize());
        DEBUG3(fprintf(stderr,"c,NUMA location of A,C: %d %d\n", inumaA, inumaC));
        DEBUG1(t1 += mysecond());
    }

    DEBUG1(t0 += mysecond());

    DEBUG3(fprintf(stderr, "gpu: single csyrk timing(s): total= %10.6f, compute= %10.6f, other= %10.6f\n", t0, t1, t0-t1));

    DEBUG1(farray[fi].t0 += t0);
    DEBUG1(farray[fi].t1 += t1);

    return;
}
