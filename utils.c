
//#include "utils.h"
#include "global.h"

#include <time.h>
#include <sys/time.h>
#include <numaif.h>
#include <numa.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>



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


int which_numa(void *var, size_t bytes) {
    int status[3];
    int ret_code;
    status[0] = -1;
    status[1] = -1;
    status[2] = -1;

    void *ptr_to_check[3];
    ptr_to_check[0] = var;
    ptr_to_check[1] = var + bytes / sizeof(void) / 2;
    ptr_to_check[2] = var + bytes / sizeof(void);

    ret_code = move_pages(0 /* self memory */, 3, ptr_to_check, NULL, status, 0);
    
    if (status[0] == 0 || status[1] == 0 || status[2] == 0) {
        return 0;
    }
    return 1;
}


int which_numa2(void *var) {
 //return 0;
 void * ptr_to_check = var;
 int status[1];
 int ret_code;
 status[0]=-1;
 ret_code=move_pages(0 /*self memory */, 1, &ptr_to_check, NULL, status, 0);
 // this print may cause extra NUMA traffic
 // if(debug) printf("Memory at %p is at numa node %d (retcode %d)\n", ptr_to_check, status[0], ret_code);
 return status[0];
}


void move_numa(void *ptr, size_t size, int target_node) {
// size in Bytes
    //printf("size in move_numa=%d, array size=%d\n",size, size/8);
    double tnuma=mysecond();
    int PAGE_SIZE = getpagesize();
    size_t rc=0;
    size_t num_pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
    int *status = malloc(num_pages*sizeof(int));
    int *nodes = malloc(num_pages*sizeof(int));
    void **page_addrs = malloc(num_pages * sizeof(void *));

    // Populate the array with page addresses
    #pragma omp parallel for
    for (size_t i = 0; i < num_pages; i++) {
        page_addrs[i] = ptr + (i * PAGE_SIZE )/sizeof(void);
        nodes[i]=target_node;
        status[i]=-1;
    }
    

#define MOVE_BULK   //OMP parallelized version is slower due to too many concurrencies when all cores are used. 
#ifdef MOVE_BULK
    rc=move_pages(0, num_pages, page_addrs, nodes, status, MPOL_MF_MOVE);
    if(rc!=0) {
        if(rc > 0) {
                fprintf(stderr, "warning: %d pages not moved\n", rc); 
                for (int i = 0; i < num_pages; i++) 
                   if (status[i] < 0) {  // Check if there's an error for this page
                       fprintf(stderr, "Page %d (at %d) not moved, error: %d %s\n", i, which_numa2(page_addrs[i]),status[i],strerror(-status[i]));
                   }
        }
        if(rc < 0) {fprintf(stderr, "error: page migration failed\n"); exit(-1);} 
    }
#else
    #pragma omp parallel
    {
        int thread_rc = 0;
        #pragma omp for
        for (size_t i = 0; i < num_pages; i += omp_get_num_threads()) {
            size_t start = i;
            size_t end = ((i + omp_get_num_threads()) < num_pages) ? (i + omp_get_num_threads()) : num_pages;
            thread_rc = move_pages(0 , end - start, &page_addrs[start], &nodes[start], &status[start], 0);
            if (thread_rc != 0) {
                #pragma omp critical
                {
                  //  if (thread_rc > 0) fprintf(stderr, "warning: %d pages not moved\n", thread_rc);
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
#endif

    free(page_addrs);
    free(nodes); ////somehow not freeing nodes makes PARSEC run much faster 250s to 235s. 
    free(status);

    tnuma=mysecond()-tnuma;
    //printf("element numa\n");
    //for (int i =0; i<size/8; i++) printf("%d %d\n",i,which_numa(ptr+i));

    if ( rc > 0) 
       DEBUG2(fprintf(stderr,"move_numa time %15.6f of %lu pages (%lu not moved)\n", tnuma, num_pages, rc));
    else
       DEBUG2(fprintf(stderr,"move_numa time %15.6f of %lu pages\n", tnuma, num_pages));
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

int get_MPI_local_rank() {
    char* pmi_rank = getenv("MPI_LOCALRANKID");
    char* mvapich_rank = getenv("MV2_COMM_WORLD_LOCAL_RANK");
    char* ompi_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (pmi_rank != NULL)
        return atoi(pmi_rank);
    else if (mvapich_rank != NULL)
        return atoi(mvapich_rank);
    else if (ompi_rank != NULL)
        return atoi(ompi_rank);
    else
        return -1;
}




void get_argv0(char **argv0) {
    char* buffer = (char *)malloc(sizeof(char) * (1024));
    strcpy(buffer, "null\0");
    FILE *fp = fopen("/proc/self/cmdline", "r");
    if (!fp) {
        perror("fopen");
        *argv0 = buffer;
        return;
    }

    int n = fread(buffer, 1, 1024, fp);
    if (n == 0) {
        perror("fread");
        *argv0 = buffer;
        return;
    }
    buffer[n-1] = '\0';
    *argv0 = buffer;
}


#define check_list_size 14
/* Check if string ends with commands to be ignored*/
int check_string(const char *str) {
    const char *check_list[check_list_size] = {
        "/bin/sh",
        "/bin/bash",
        "lscpu",
        "bin/ssh",
        "hostname",
        //"awk", "sed", "grep", "lscpu", "mktemp", "rm", "mv",
        "ibrun",
        "mpirun",
        "mpirun_rsh",
        "mpiexec",
        "mpiexec.hydra",
        "numactl",
        "srun",
        "hydra_bstrap_proxy",
        "hydra_pmi_proxy"
    };

    for (int i = 0; i < check_list_size; i++) {
        size_t len = strlen(check_list[i]);
        if (strlen(str) >= len && strcmp(str + strlen(str) - len, check_list[i]) == 0) {
            //printf("The string %s ends with %s\n", str, check_list[i]);
            return 1;
        }
    }
    return 0;
}

