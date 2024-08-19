
#define _GNU_SOURCE
//#include "utils.h"
#include "global.h"

#include <time.h>
#include <sys/time.h>
#include <numaif.h>
#include <numa.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <limits.h>




double scilib_second()
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

double scilib_second2()
{
    struct timespec measure;

    // Get the current time as the start time
    clock_gettime(CLOCK_MONOTONIC, &measure);

    // Return the elapsed time in seconds
    return (double)measure.tv_sec + (double)measure.tv_nsec * 1e-9;
}

double scilib_second_() {return scilib_second();}
double scilib_second2_() {return scilib_second2();}


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
    double tnuma=scilib_second();
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
      

#define MOVE_BULK   //OMP parallelized version is slower (230s->250s) due to too many concurrencies when all cores are used. 
#ifdef MOVE_BULK
    rc=move_pages(0, num_pages_plus, page_addrs, nodes, status, MPOL_MF_MOVE);
    if(rc!=0) {
        if(rc > 0 && scilib_debug >=3) {
                fprintf(stderr, "warning: %d pages not moved\n", rc); 
                for (int i = 0; i < num_pages; i++) 
                   if (status[i] < 0) {  // Check if there's an error for this page
                       fprintf(stderr, "Page %d (at %d) not moved, error: %d %s\n", i, which_numa(page_addrs[i],0),status[i],strerror(-status[i]));
                       exit(-1);
                   }
        }
        if(rc < 0) {fprintf(stderr, "error: page migration failed\n"); exit(-1);} 
    }
#else
    #pragma omp parallel   //need to double check this part
    {
        int thread_rc = 0;
        #pragma omp for
        for (size_t i = 0; i < num_pages_plus; i += omp_get_num_threads()) {
            size_t start = i;
            size_t end = ((i + omp_get_num_threads()) < num_pages_plus) ? (i + omp_get_num_threads()) : num_pages_plus;
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
    free(nodes);  
    free(status);

    tnuma=scilib_second()-tnuma;
    //printf("element numa\n");
    //for (int i =0; i<size; i++) printf("%d %d\n",i,which_numa(ptr+i,1));

    if ( rc > 0) 
       DEBUG2(fprintf(stderr,"move_page time %15.6f of %lu pages (%d not moved)\n", tnuma, num_pages, rc));
    else
       DEBUG2(fprintf(stderr,"move_page time %15.6f of %lu pages\n", tnuma, num_pages));
    return;
}



void get_exe_path(char **path){
    char *exe_path=malloc(PATH_MAX);
//    char exe_path[PATH_MAX];  //destroyed after call
    ssize_t len = readlink("/proc/self/exe", exe_path, PATH_MAX-1);
    if (len != -1) {
        exe_path[len] = '\0';
        //printf("Executable path: %s,  len: %zd\n", exe_path, len);
       *path = exe_path;
    } else {
        perror("Failed to get executable path");
        free(exe_path);
        *path = NULL;
        return;
    }
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


#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))
/* Check if string ends with commands to be ignored*/
int check_string(const char *str) {
    const char *exe_list[] = {
        "ibrun",
        "mpirun",
        "orterun",
        "orted",
        "mpirun_rsh",
        "mpiexec",
        "mpiexec.hydra",
        "srun",
        "hydra_bstrap_proxy",
        "hydra_pmi_proxy",
        "numactl",
        "pip",
        "pip3",
        "virtualenv",
// profilers
        "map",
        "scorep",
        "nvprof",
        "nsys",
        "ncu",
    };

    const char *dir_list[] = {
        "/bin",
        "/usr",
        "/sbin",
    };

    // Check if the program path starts with any in the dir_list
    //fprintf(stderr, "str=%s\n", str);

    for (int i = 0; i < ARRAY_SIZE(dir_list); i++) {
        size_t len = strlen(dir_list[i]);
        //fprintf(stderr, "dir_list=%s\n", dir_list[i]);
        if (strlen(str) >= len && strncmp(str, dir_list[i], len) == 0) {
         //     fprintf(stderr, "--skip due to dir\n");
            return 1;
        }
    }
    // Check if the executable name matches any in the list
    for (int i = 0; i < ARRAY_SIZE(exe_list); i++) {
        size_t len = strlen(exe_list[i]);
        //fprintf(stderr, "exe_list=%s\n", exe_list[i]);
        if (strlen(str) >= len && strcmp(str + strlen(str) - len, exe_list[i]) == 0) {
        //      fprintf(stderr, "--skip due to exe\n");
            return 1;
        }
    }

    return 0;
}



#include <assert.h>
#include <string.h>
char** str_split(char* a_str, const char a_delim)
{
    char** result    = 0;
    size_t count     = 0;
    char* tmp        = a_str;
    char* last_comma = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;

    /* Count how many elements will be extracted. */
    while (*tmp)
    {
        if (a_delim == *tmp)
        {
            count++;
            last_comma = tmp;
        }
        tmp++;
    }

    /* Add space for trailing token. */
    count += last_comma < (a_str + strlen(a_str) - 1);

    /* Add space for terminating null string so caller
       knows where the list of returned strings ends. */
    count++;

    result = malloc(sizeof(char*) * count);

    if (result)
    {
        size_t idx  = 0;
        char* token = strtok(a_str, delim);

        while (token)
        {
            assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        assert(idx == count - 1);
        *(result + idx) = 0;
    }

    return result;
}

int in_str(const char* s1, char** s2) {
    if (s2 == NULL || s1 == NULL) return 0;  
    
    for (int i = 0; s2[i] != NULL; i++) {
        if (strcmp(s2[i], s1) == 0) {
            return 1;  // Found a match
        }
    }
    
    return 0;  // No match found
}



/*  Experimental  */

#include <errno.h>
#include <unistd.h>  // For sysconf
#include <dlfcn.h> 
#include <fcntl.h>

//size_t pagesize = sysconf(_SC_PAGESIZE);
#define S_1G 1024*1024*1024*2
#define S_1M 1024*1024
#define S_64K 65536
#define S_4K 1024
#define NALIGN S_64K 
#define BAR  1 //S_1M*10
size_t nalign=NALIGN;
size_t bar=BAR;
void* xmalloc(size_t size) {
    static void* (*real_malloc)() = NULL; 
    void *ptr = NULL;
    if (!real_malloc)  real_malloc = dlsym(RTLD_NEXT, "malloc") ;

    if ( 0 ||  size < bar) return real_malloc(size);

    size = ( size/nalign + 1 ) * nalign;
    int result = posix_memalign(&ptr, nalign, size);
    return ptr;
}

