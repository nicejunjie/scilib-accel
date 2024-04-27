

#include "utils.h"


double mysecond()
{
/* struct timeval { long        tv_sec;
            long        tv_usec;        };

struct timezone { int   tz_minuteswest;
             int        tz_dsttime;      };     */

        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

double mysecond2()
{
    struct timespec measure;

    // Get the current time as the start time
    clock_gettime(CLOCK_MONOTONIC, &measure);

    // Return the elapsed time in seconds
    return (double)measure.tv_sec + (double)measure.tv_nsec * 1e-9;
}

double mysecond_() {return mysecond();}
double mysecond2_() {return mysecond2();}





int which_numa(double *var) {
 void * ptr_to_check = var;
 int status[1];
 int ret_code;
 status[0]=-1;
 ret_code=move_pages(0 /*self memory */, 1, &ptr_to_check, NULL, status, 0);
 // this print may cause extra NUMA traffic
 // if(debug) printf("Memory at %p is at numa node %d (retcode %d)\n", ptr_to_check, status[0], ret_code);
 return status[0];
}


void move_numa(double *ptr, size_t size, int target_node) {
// size in Bytes
    //printf("size in move_numa=%d, array size=%d\n",size, size/8);
    double tnuma=mysecond();
    int PAGE_SIZE = getpagesize();
    int rc=0;
    size_t num_pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
    int *status = malloc(num_pages*sizeof(int));
    int *nodes = malloc(num_pages*sizeof(int));
    void **page_addrs = malloc(num_pages * sizeof(void *));

    // Populate the array with page addresses
    #pragma omp parallel for
    for (size_t i = 0; i < num_pages; i++) {
        page_addrs[i] = ptr + (i * PAGE_SIZE / sizeof(double));
        nodes[i]=target_node;
        status[i]=-1;
    }
//    rc=move_pages(0 /*self memory*/, num_pages, page_addrs, nodes, status, 0);
/*
    if(rc!=0) {
        if(rc > 0) fprintf(stderr, "warning: %d pages not moved\n", rc); 
        if(rc < 0) {fprintf(stderr, "error: page migration failed\n"); exit(-1);} 
    }
*/

    #pragma omp parallel
    {
        int thread_rc = 0;
        #pragma omp for
        for (size_t i = 0; i < num_pages; i += omp_get_num_threads()) {
            size_t start = i;
            size_t end = ((i + omp_get_num_threads()) < num_pages) ? (i + omp_get_num_threads()) : num_pages;
            thread_rc = move_pages(0 /*self memory*/, end - start, &page_addrs[start], &nodes[start], &status[start], 0);
            if (thread_rc != 0) {
                #pragma omp critical
                {
                    if (thread_rc > 0) fprintf(stderr, "warning: %d pages not moved\n", thread_rc);
                    if (thread_rc < 0) {
                        fprintf(stderr, "error: page migration failed\n");
                        exit(-1);
                    }
                }
            }
        }
        #pragma omp critical
        {
            rc += thread_rc;
        }
    }

    free(page_addrs);

    tnuma=mysecond()-tnuma;
    //printf("element numa\n");
    //for (int i =0; i<size/8; i++) printf("%d %d\n",i,which_numa(ptr+i));
    printf("move_numa time %15.6f of %lu pages\n", tnuma, num_pages);
    //mtime_dmove+=tnuma;
    return;
}




int check_MPI() {
    char* pmi_rank = getenv("PMI_RANK");
    //char* pmix_rank = getenv("MPIX_RANK");
    char* mvapich_rank = getenv("MV2_COMM_WORLD_RANK");
    char* ompi_rank = getenv("OMPI_COMM_WORLD_RANK");
    //char* slurm_rank = getenv("SLURM_PROCID");

    if (pmi_rank != NULL  || mvapich_rank != NULL || ompi_rank != NULL )
        return 1;
    else
        return 0;
}

