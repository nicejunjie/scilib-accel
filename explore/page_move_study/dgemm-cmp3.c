
#define _GNU_SOURCE
#include <dlfcn.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "nvpl_blas_cblas.h"
#include <errno.h>
#include <cuda.h> 


double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;  // returns seconds
}


#define MEMORY_ALIGNMENT  65536
#ifdef MEM_ALIGN
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )
#else
#define ALIGN_UP(x,size) x 
#endif

void* xmalloc(size_t size) {
    void *ptr = NULL;

    // Align to MEMORY_ALIGNMENT bytes (4KB in this case)
    int result = posix_memalign(&ptr, MEMORY_ALIGNMENT, size);
    
    if (result != 0) {
        errno = result; // Set errno if allocation fails
        return NULL;
    }

    return ptr;
}

/*
void* ymalloc(size_t size){
  static int inmalloc=0;
  static void* (*original_malloc)(size_t);
  inmalloc++; 
  void* result;
  if (!original_malloc) original_malloc = dlsym(RTLD_NEXT, "malloc");
  if(inmalloc==0) result=malloc_managed(size);
  else  result=original_malloc(size);
  inmalloc--; 
  return result;
}
*/



#include <numaif.h>
#include <numa.h>
#include <unistd.h>

int which_numa(void *ptr, size_t bytes) {
    int ret_code;
    int status[3];
    void *ptr_to_check[3];
    char *char_ptr = (char*)ptr;
    int n = 3;

    for (int i = 0; i < 3; i++) status[i] = -1;

    ptr_to_check[0] = char_ptr;
    ptr_to_check[1] = char_ptr + bytes / 2;
    ptr_to_check[2] = char_ptr + bytes - 1;

    if ( bytes == 0 ) n = 1 ;  
    else if ( bytes < 3 ) n = bytes; 

    ret_code = move_pages(0 /* self memory */, n, ptr_to_check, NULL, status, 0);
    if(ret_code != 0) fprintf(stderr, "issue in which_numa\n");
/*
#define MOVE_PAGES 279  // syscall number for move_pages
    ret_code = syscall(MOVE_PAGES, 0, 3, ptr_to_check, NULL, status, 0);
*/
    if (status[0] == 0 || status[1] == 0 || status[2] == 0) return 0;
    return 1;
}

void move_numa(void *ptr, size_t size, int target_node) {
// size in Bytes
    //printf("size in move_numa=%d, array size=%d\n",size, size/8);
    double tnuma=get_time();
    int PAGE_SIZE = getpagesize();
    int rc=0;
    size_t num_pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
    size_t num_pages_plus = num_pages + 1; //account for the last page
    int *status = malloc(num_pages_plus*sizeof(int));
    int *nodes = malloc(num_pages_plus*sizeof(int));
    void **page_addrs = malloc(num_pages_plus * sizeof(void *));

    char * char_ptr = ptr; 
    // Populate the array with page addresses
    #pragma omp parallel for
    for (size_t i = 0; i < num_pages; i++) {
        page_addrs[i] = char_ptr + i * PAGE_SIZE ;
        nodes[i]=target_node;
        status[i]=-1;
    }
      
// check the last byte 
    page_addrs[num_pages] = char_ptr + size -1 ;
    nodes[num_pages] = target_node;
    status[num_pages] = -1;
      

    rc=move_pages(0, num_pages_plus, page_addrs, nodes, status, MPOL_MF_MOVE);
    if(rc!=0) {
        if(rc > 0) {
                fprintf(stderr, "warning: %d pages not moved\n", rc); 
                for (int i = 0; i < num_pages; i++) 
                   if (status[i] < 0) {  // Check if there's an error for this page
                       fprintf(stderr, "Page %d (at %d) not moved, error: %d %s\n", i, which_numa(page_addrs[i],0),status[i],strerror(-status[i]));
                       exit(-1);
                   }
        }
        if(rc < 0) {fprintf(stderr, "error: page migration failed\n"); exit(-1);} 
    }
    free(page_addrs);
    free(nodes);  
    free(status);

    tnuma=get_time()-tnuma;
    //printf("element numa\n");
    //for (int i =0; i<size; i++) printf("%d %d\n",i,which_numa(ptr+i,1));

    if ( rc > 0) 
       fprintf(stderr,"move_page time %15.6f of %lu pages (%d not moved)\n", tnuma, num_pages, rc);
    else
       fprintf(stderr,"move_page time %15.6f of %lu pages\n", tnuma, num_pages);
    return;
}




int main(int argc, char *argv[]) {
    if (argc != 5 && argc != 7) {
        printf("Usage: %s M N K iteration\n", argv[0]);
        printf("or Usage: %s M N K iter1 iter2 iter3\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int Iter1, Iter2, Iter3;
    if (argc == 5) {
       Iter1 = atoi(argv[4]);
       Iter2 = Iter1;
       Iter3 = Iter1;
    }
    else  {
       Iter1 = atoi(argv[4]);
       Iter2 = atoi(argv[5]);
       Iter3 = atoi(argv[6]);
    }


    // Calculate total data size for bandwidth calculation
    double dsizeA = ((double)M * (double)K) * sizeof(double);
    double dsizeB = ((double)K * (double)N) * sizeof(double);
    double dsizeC = ((double)M * (double)N) * sizeof(double);
    double total_data_size = dsizeA + dsizeB + dsizeC; 

    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Matrix size: A %10.2f MB, B %10.2f MB, C %10.2f MB, Total %10.2f MB\n", dsizeA/1e6, dsizeB/1e6, dsizeC/1e6, total_data_size/1e6);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = 1.0;
    const double beta = 0.0;

    // Allocate host memory using malloc
    size_t sizeA = M * K * sizeof(double);
    size_t sizeB = K * N * sizeof(double);
    size_t sizeC = M * N * sizeof(double);

    double *A_ua = (double*)malloc(sizeA + MEMORY_ALIGNMENT);
    double *B_ua = (double*)malloc(sizeB + MEMORY_ALIGNMENT);
    double *C_ua = (double*)malloc(sizeC + MEMORY_ALIGNMENT);
    double *A = (double*) ALIGN_UP(A_ua, MEMORY_ALIGNMENT);
    double *B = (double*) ALIGN_UP(B_ua, MEMORY_ALIGNMENT);
    double *C = (double*) ALIGN_UP(C_ua, MEMORY_ALIGNMENT);
/*
    double *A = (double*)malloc_managed(sizeA);
    double *B = (double*)malloc_managed(sizeB);
    double *C = (double*)malloc_managed(sizeC);
*/


    // Initialize matrices
    for (int i = 0; i < M * K; i++) {
        A[i] = 1.0;
        A_ua[i] = 1.0;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = 2.0;
        B_ua[i] = 2.0;
    }
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0;
        C_ua[i] = 0.0;
    }

   
   double start_time, end_time;


   // CPU BLAS
    printf("\n-------------------------------\n");
    for(int i=0; i<Iter1; i++) { 
        // 1.2 CPU dgemm with malloc memory
        start_time = get_time();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, M);
        end_time = get_time();
        printf("  iteration %3d,  CPU dgemm time         : %10.3f ms,  numa A B C: %d %d %d\n", i, (end_time - start_time)*1e3, which_numa(A,sizeA), which_numa(B,sizeB), which_numa(C,sizeC));

    }

#ifdef MOVE_NUMA
   move_numa(A, sizeA, 1);
   move_numa(B, sizeB, 1);
   move_numa(C, sizeC, 1);
#endif
    
   // GPU BLAS
 /*  advise doesn't change anything. */
#ifdef MEM_ADVISE
    cudaMemAdvise(A, sizeA, cudaMemAdviseSetPreferredLocation, 0); // Prefer GPU for A
    cudaMemAdvise(B, sizeB, cudaMemAdviseSetPreferredLocation, 0); // Prefer GPU for B
    cudaMemAdvise(C, sizeC, cudaMemAdviseSetPreferredLocation, 0); // Prefer GPU for C
#endif

    printf("\n-------------------------------\n");
    for(int i=0; i<Iter2; i++) { 
        // 1.2 CPU dgemm with malloc memory
        start_time = get_time();
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M);
        cudaDeviceSynchronize();
        end_time = get_time();
        printf("  iteration %3d, cublasdgemm time        : %10.3f ms,  numa A B C: %d %d %d\n", i, (end_time - start_time)*1e3, which_numa(A,sizeA), which_numa(B,sizeB), which_numa(C,sizeC));
    }

   // CPU BLAS
    printf("\n-------------------------------\n");
    for(int i=0; i<Iter3; i++) { 
        // 1.2 CPU dgemm with malloc memory
        start_time = get_time();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, M);
        end_time = get_time();
        printf("  iteration %3d,  CPU dgemm time         : %10.3f ms,  numa A B C: %d %d %d\n", i, (end_time - start_time)*1e3, which_numa(A,sizeA), which_numa(B,sizeB), which_numa(C,sizeC));

    }


    printf("\n-------------------------------\n");


    // Clean up
    free(A_ua);
    free(B_ua);
    free(C_ua);


    cublasDestroy(handle);

    return 0;
}
