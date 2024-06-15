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

    farray[fi].t0 -= mysecond();

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
         farray[fi].t1 -= mysecond();
         orig_f(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
         double ts = mysecond();
         farray[fi].t1 += ts;
         farray[fi].t0 += ts;
         return;
    }
    fprintf(stderr,"gpu: dgemm args: transa=%c, transb=%c, m=%d, n=%d, k=%d, alpha=%.1f, lda=%d, ldb=%d, beta=%.1f, ldc=%d\n",
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

#define SUBCOPY
#define SUBTMP  //only copy a submatrix      
#ifdef SUBCOPY 
    bool subA=false, subB=false, subC=false;
    double *tempA, *tempB, *tempC;

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
  
/*
    lda_gpu=*lda;
    ldb_gpu=*ldb;
    ldc_gpu=*ldc;
    sizeA_gpu=sizeA;
    sizeB_gpu=sizeB;
    sizeC_gpu=sizeC;
*/
    CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeA_gpu, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_B, sizeB_gpu, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_C, sizeC_gpu, stream));
   
    if (subA) {
#ifdef SUBTMP
      tempA = (double*)malloc(sizeA_gpu);
#pragma omp parallel for
      for (int j = 0; j < A_Op_col; j++) 
        memcpy(tempA + j * lda_gpu, (double*)A + j * *lda, A_Op_row * size_type);
      CUDA_CHECK(cudaMemcpyAsync(d_A, tempA, sizeA_gpu, cudaMemcpyHostToDevice, stream));
#else 
      for (int j = 0; j < A_Op_col; j++) 
        CUDA_CHECK(cudaMemcpyAsync(d_A + j * lda_gpu, (double*)A + j * *lda, A_Op_row * size_type,
                                 cudaMemcpyHostToDevice, stream));
#endif
    }
    else 
      CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeA, cudaMemcpyHostToDevice, stream));

  
    if (subB) {
#ifdef SUBTMP
      tempB = (double*)malloc(sizeB_gpu);
#pragma omp parallel for
      for (int j = 0; j < B_Op_col; j++) 
        memcpy(tempB + j * ldb_gpu, (double*)B + j * *ldb, B_Op_row * size_type);
      CUDA_CHECK(cudaMemcpyAsync(d_B, tempB, sizeB_gpu, cudaMemcpyHostToDevice, stream));
#else
      for (int j = 0; j < B_Op_col; j++) 
        CUDA_CHECK(cudaMemcpyAsync(d_B + j * ldb_gpu, (double*)B + j * *ldb, B_Op_row * size_type,
                                 cudaMemcpyHostToDevice, stream));
#endif
    }
    else 
      CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeB, cudaMemcpyHostToDevice, stream));

  
    if (subC) {
#ifdef SUBTMP
      tempC = (double*)malloc(sizeC_gpu);
#pragma omp parallel for
      for (int j = 0; j < C_Op_col; j++) 
        memcpy(tempC + j * ldc_gpu, (double*)C + j * *ldc, C_Op_row * size_type);
      CUDA_CHECK(cudaMemcpyAsync(d_C, tempC, sizeC_gpu, cudaMemcpyHostToDevice, stream));
#else
      for (int j = 0; j < C_Op_col; j++) 
        CUDA_CHECK(cudaMemcpyAsync(d_C + j * ldc_gpu,(double*)C + j * *ldc,  C_Op_row * size_type,
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

    farray[fi].t1 -= mysecond();
    CUBLAS_CHECK(cublasDgemm(handle, transA, transB, *m, *n, *k, alpha, d_A, lda_gpu, d_B, ldb_gpu, beta, d_C, ldc_gpu));
    CUDA_CHECK(cudaDeviceSynchronize());
    farray[fi].t1 += mysecond();

    if(subC) {
#ifdef SUBTMP
      CUDA_CHECK(cudaMemcpyAsync(tempC, d_C, sizeC_gpu, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaDeviceSynchronize());
#pragma omp parallel for
      for (int j = 0; j < C_Op_col; j++) 
        memcpy((double*)C + j * *ldc, tempC + j * ldc_gpu, C_Op_row * size_type);
      free(tempC);
#else 
      for (int j = 0; j < C_Op_col; j++) 
        CUDA_CHECK(cudaMemcpyAsync((double*)C + j * *ldc, d_C + j * ldc_gpu, C_Op_row * size_type,
                                 cudaMemcpyDeviceToHost, stream));
#endif
    }
    else 
      CUDA_CHECK(cudaMemcpyAsync(C, d_C, sizeC, cudaMemcpyDeviceToHost, stream));

#else 
    CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeA, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_B, sizeB, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&d_C, sizeC, stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpyAsync(d_A, A, sizeA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B, sizeB, cudaMemcpyHostToDevice, stream));
//    if( beta_abs > 1.0e-8 )  bug if gemm on a submatrix
    CUDA_CHECK(cudaMemcpyAsync(d_C, C, sizeC, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    farray[fi].t1 -= mysecond();
    CUBLAS_CHECK(cublasDgemm(handle, transA, transB, *m, *n, *k, alpha, d_A, *lda, d_B, *ldb, beta, d_C, *ldc));
    CUDA_CHECK(cudaDeviceSynchronize());
    farray[fi].t1 += mysecond();
    CUDA_CHECK(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));

#endif
    CUDA_CHECK(cudaDeviceSynchronize());
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
/*
#else  //experiment advise location for cuda managed memory
    int device_id;
    cudaGetDevice(&device_id);

    cudaMemAdvise(A, sizeA, cudaMemAdviseSetPreferredLocation, device_id);
    cudaMemAdvise(B, sizeB, cudaMemAdviseSetPreferredLocation, device_id);
    cudaMemAdvise(C, sizeC, cudaMemAdviseSetPreferredLocation, device_id);
*/
#endif


    farray[fi].t1 -= mysecond();
    CUBLAS_CHECK(cublasDgemm(handle, transA, transB, *m, *n, *k, alpha, A, *lda, B, *ldb, beta, C, *ldc));
    CUDA_CHECK(cudaDeviceSynchronize());
    farray[fi].t1 += mysecond();
#endif


    farray[fi].t0 += mysecond();

    return;
}




