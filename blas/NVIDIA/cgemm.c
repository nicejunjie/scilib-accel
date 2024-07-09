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

    int size_type = sizeof(cuFloatComplex); //for complex
    size_t sizeA = (transa[0] == 'N'||transa[0] == 'n') ? ((*k) * (*lda)) : ((*m) * (*lda));
    size_t sizeB = (transb[0] == 'N'||transb[0] == 'n') ? ((*n) * (*ldb)) : ((*k) * (*ldb));
    size_t sizeC = (*n) * (*ldc);
    sizeA *= size_type;
    sizeB *= size_type;
    sizeC *= size_type;
    double cgemm_mem_size_mb = ((double)sizeA+(double)sizeB+(double)sizeC) / 1024.0 / 1024.0;
    cuFloatComplex *beta2=(cuFloatComplex *)beta;
    double beta_abs = cuCabsf( *beta2);
    int ic = (beta_abs > 0.00000001) ? 2:1; 

    if(avgn<scilib_matrix_offload_size)  {
         DEBUG2(fprintf(stderr, "cpu: cgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, lda=%d, ldb=%d, ldc=%d, alpha=(%.1e, %.1e), beta=(%.1e, %.1e)\n",
           *transa, *transb, *m, *n, *k, *lda, *ldb, *ldc, crealf(*(float complex*)alpha), cimagf(*(float complex*)alpha), crealf(*(float complex*)beta), cimagf(*(float complex*)beta)));

         if (!orig_f) orig_f = farray[fi].fptr;
         DEBUG1(t1 -= mysecond());
         orig_f(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
         double ts;
         DEBUG1(ts = mysecond());
         DEBUG1(t1 += ts);
         DEBUG1(t0 += ts);

         DEBUG1(farray[fi].t0 += t0);
         DEBUG1(farray[fi].t1 += t1);

         return;
    }

    DEBUG2(fprintf(stderr, "gpu: cgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, lda=%d, ldb=%d, ldc=%d, alpha=(%.1e, %.1e), beta=(%.1e, %.1e)\n",
        *transa, *transb, *m, *n, *k, *lda, *ldb, *ldc, crealf(*(float complex*)alpha), cimagf(*(float complex*)alpha), crealf(*(float complex*)beta), cimagf(*(float complex*)beta)));
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
    else if (transa[0] == 'H' || transa[0] == 'h') 
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


//#ifdef GPUCOPY
if (scilib_offload_mode==1) {
    cuFloatComplex *d_A, *d_B, *d_C;

//#define SUBCOPY  // only copy the submatrix
//#define SUBTMP  
////memcpy the submatrix to a tmp matrix and cudamemcpy to GPU, 
//otherwise copy one column at a time directly to GPU.
//MuST:  80s with all copy,  106s without subtmp submatrix copy, 67s with subtmp
#ifdef SUBCOPY 
    bool subA=false, subB=false, subC=false;
    cuFloatComplex *tempA, *tempB, *tempC;

    int A_Op_row, A_Op_col, lda_gpu, B_Op_row, B_Op_col, ldb_gpu, C_Op_row, C_Op_col, ldc_gpu;
    size_t sizeA_gpu, sizeB_gpu, sizeC_gpu;

    if (transa[0] == 'N' || transa[0] == 'n') {A_Op_row=*m; A_Op_col=*k;}
    else {A_Op_row=*k; A_Op_col=*m;}
    if(*lda == A_Op_row)  {
        lda_gpu = *lda;
        sizeA_gpu = sizeA;
    }
    else {
        lda_gpu=A_Op_row;
        subA = true;
        sizeA_gpu = A_Op_row*A_Op_col*size_type;
    }
  
    if (transb[0] == 'N' || transb[0] == 'n') {B_Op_row=*k; B_Op_col=*n;}
    else {B_Op_row=*n; B_Op_col=*k;}
    if(*ldb == B_Op_row)  {
        ldb_gpu = *ldb;
        sizeB_gpu = sizeB;
    }
    else {
        ldb_gpu=B_Op_row;
        subB = true;
        sizeB_gpu = B_Op_row*B_Op_col*size_type;
    }
  
    C_Op_row=*m; C_Op_col=*n;
    if(*ldc == C_Op_row)  {
        ldc_gpu = *ldc;
        sizeC_gpu = sizeC;
    }
    else {
        ldc_gpu=C_Op_row;
        subC = true;
        sizeC_gpu = C_Op_row*C_Op_col*size_type;
    }
  
    CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeA_gpu, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_B, sizeB_gpu, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_C, sizeC_gpu, stream));
   

    if (subA) {
#ifdef SUBTMP
      tempA = (cuFloatComplex*)malloc(sizeA_gpu);
#pragma omp parallel for
      for (int j = 0; j < A_Op_col; j++) 
        memcpy(tempA + j * lda_gpu, (cuFloatComplex*)A + j * *lda, A_Op_row * size_type);
      CUDA_CHECK(cudaMemcpyAsync(d_A, tempA, sizeA_gpu, cudaMemcpyHostToDevice, stream));
#else 
#pragma omp parallel for
      for (int j = 0; j < A_Op_col; j++) 
        CUDA_CHECK(cudaMemcpyAsync(d_A + j * lda_gpu, (cuFloatComplex*)A + j * *lda, A_Op_row * size_type,
                                 cudaMemcpyHostToDevice, stream));
#endif
    }
    else 
      CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeA, cudaMemcpyHostToDevice, stream));

    if (subB) {
#ifdef SUBTMP
      tempB = (cuFloatComplex*)malloc(sizeB_gpu);
#pragma omp parallel for
      for (int j = 0; j < B_Op_col; j++) 
        memcpy(tempB + j * ldb_gpu, (cuFloatComplex*)B + j * *ldb, B_Op_row * size_type);
      CUDA_CHECK(cudaMemcpyAsync(d_B, tempB, sizeB_gpu, cudaMemcpyHostToDevice, stream));
#else
      for (int j = 0; j < B_Op_col; j++) 
        CUDA_CHECK(cudaMemcpyAsync(d_B + j * ldb_gpu, (cuFloatComplex*)B + j * *ldb, B_Op_row * size_type,
                                 cudaMemcpyHostToDevice, stream));
#endif
    }
    else 
      CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeB, cudaMemcpyHostToDevice, stream));

    if (subC) {
#ifdef SUBTMP
      tempC = (cuFloatComplex*)malloc(sizeC_gpu);
#pragma omp parallel for
      for (int j = 0; j < C_Op_col; j++) 
        memcpy(tempC + j * ldc_gpu, (cuFloatComplex*)C + j * *ldc, C_Op_row * size_type);
      CUDA_CHECK(cudaMemcpyAsync(d_C, tempC, sizeC_gpu, cudaMemcpyHostToDevice, stream));
#else
      for (int j = 0; j < C_Op_col; j++) 
        CUDA_CHECK(cudaMemcpyAsync(d_C + j * ldc_gpu,(cuFloatComplex*)C + j * *ldc,  C_Op_row * size_type,
                                   cudaMemcpyHostToDevice, stream));
#endif
    }
    else 
      CUDA_CHECK(cudaMemcpyAsync(d_C, C, sizeC, cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaDeviceSynchronize());
#ifdef SUBTMP
    if (subA) free(tempA);
    if (subB) free(tempB);
#endif

    DEBUG1(t1 -= mysecond());
    CUBLAS_CHECK(_CUBLASCGEMM(handle, transA, transB, *m, *n, *k, alpha, d_A, lda_gpu, d_B, ldb_gpu, beta, d_C, ldc_gpu));
    CUDA_CHECK(cudaDeviceSynchronize());
    DEBUG1(t1 += mysecond());

    if(subC) {
#ifdef SUBTMP
      CUDA_CHECK(cudaMemcpyAsync(tempC, d_C, sizeC_gpu, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaDeviceSynchronize());
#pragma omp parallel for
      for (int j = 0; j < C_Op_col; j++) 
        memcpy((cuFloatComplex*)C + j * *ldc, tempC + j * ldc_gpu, C_Op_row * size_type);
      free(tempC);
#else 
      for (int j = 0; j < C_Op_col; j++) 
        CUDA_CHECK(cudaMemcpyAsync((cuFloatComplex*)C + j * *ldc, d_C + j * ldc_gpu, C_Op_row * size_type,
                                 cudaMemcpyDeviceToHost, stream));
#endif
    }
    else 
      CUDA_CHECK(cudaMemcpyAsync(C, d_C, sizeC, cudaMemcpyDeviceToHost, stream));

#else  // not copying submatrix but copy all
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
#endif
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFreeAsync(d_A, stream));
    CUDA_CHECK(cudaFreeAsync(d_B, stream));
    CUDA_CHECK(cudaFreeAsync(d_C, stream));

} //#else  //not GPUCPOY
else {

//#ifdef AUTO_NUMA
    if(scilib_offload_mode == 3){
       int inumaA=which_numa(A, sizeA);
       int inumaB=which_numa(B, sizeB);
       int inumaC=which_numa(C, sizeC);
       if ( inumaA == 0 ) move_numa(A, (size_t)sizeA, NUMA_HBM);
       if ( inumaB == 0 ) move_numa(B, (size_t)sizeB, NUMA_HBM);
       if ( inumaC == 0 ) move_numa(C, (size_t)sizeC, NUMA_HBM);
    }
/*
#else  //experiment advise location for cuda managed memory
  
    static int device_id=-1;
    if(device_id == -1) cudaGetDevice(&device_id);

    cudaMemAdvise(A, sizeA, cudaMemAdviseSetPreferredLocation, device_id);
    cudaMemAdvise(B, sizeB, cudaMemAdviseSetPreferredLocation, device_id);
    cudaMemAdvise(C, sizeC, cudaMemAdviseSetPreferredLocation, device_id);
#endif
*/

    DEBUG1(t1 -= mysecond());
    CUBLAS_CHECK(_CUBLASCGEMM(handle, transA, transB, *m, *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc));
    CUDA_CHECK(cudaDeviceSynchronize());
    DEBUG1(t1 += mysecond());
}  //#endif

    DEBUG1(t0 += mysecond());

    DEBUG3(fprintf(stderr, "single cgemm timing(s): total= %10.6f, compute= %10.6f, other= %10.6f\n", t0, t1, t0-t1));

    DEBUG1(farray[fi].t0 += t0);
    DEBUG1(farray[fi].t1 += t1);

    return;
}

