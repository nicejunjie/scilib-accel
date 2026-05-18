
#define _GNU_SOURCE
//#include "utils.h"
#include "global.h"

#include <time.h>
#include <sys/time.h>
#include <numaif.h>
#include <numa.h>
#include <sys/mman.h>
#include <errno.h>
#include <pthread.h>
#include <omp.h>
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


// --- Pointer cache for tracking already-migrated regions ---
#define MIGRATE_CACHE_SIZE 4096
static struct { void *ptr; size_t size; } migrate_cache[MIGRATE_CACHE_SIZE];
static int migrate_cache_count = 0;

static int is_migrated(void *ptr, size_t size) {
    for (int i = 0; i < migrate_cache_count; i++) {
        if (migrate_cache[i].ptr == ptr && migrate_cache[i].size >= size)
            return 1;
    }
    return 0;
}

static void add_to_migrate_cache(void *ptr, size_t size) {
    for (int i = 0; i < migrate_cache_count; i++) {
        if (migrate_cache[i].ptr == ptr) {
            if (size > migrate_cache[i].size) migrate_cache[i].size = size;
            return;
        }
    }
    if (migrate_cache_count < MIGRATE_CACHE_SIZE) {
        migrate_cache[migrate_cache_count].ptr = ptr;
        migrate_cache[migrate_cache_count].size = size;
        migrate_cache_count++;
    }
}

int which_numa(void *ptr, size_t bytes) {
    if (is_migrated(ptr, bytes)) return 1;

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

    if (status[0] == 0 || status[1] == 0 || status[2] == 0) return 0;
    return 1;
}

#include "nvidia.h"
#include "gpu_migrate.h"

// One-time page size lookup (cached). getpagesize() is a syscall on some libcs.
static int page_size_cached(void) {
    static int ps = 0;
    if (!ps) ps = getpagesize();
    return ps;
}

// Align [ptr, ptr+size) outward to whole pages.
static void page_align(const void *ptr, size_t size,
                       char **aligned_ptr_out, size_t *aligned_size_out) {
    int ps = page_size_cached();
    char *aligned_ptr = (char *)((unsigned long)ptr & ~(unsigned long)(ps - 1));
    size_t aligned_size = (size_t)((char *)ptr + size - aligned_ptr);
    aligned_size = (aligned_size + ps - 1) & ~(size_t)(ps - 1);
    *aligned_ptr_out = aligned_ptr;
    *aligned_size_out = aligned_size;
}

// mbind a single contiguous range, optionally split into N parallel chunks
// when there are enough pages to make threading worthwhile.
static void mbind_parallel_chunks(char *aligned_ptr, size_t aligned_size,
                                  int target_node, int flags) {
    int ps = page_size_cached();
    unsigned long nodemask = 1UL << target_node;
    size_t num_pages = aligned_size / ps;
    int nthreads = omp_get_max_threads();
    if (nthreads > 1 && aligned_size >= (size_t)ps * nthreads) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            size_t ppt = num_pages / nt;
            size_t sp = tid * ppt;
            size_t ep = (tid == nt - 1) ? num_pages : sp + ppt;
            size_t off = sp * ps;
            size_t cs  = (ep - sp) * (size_t)ps;
            if (off + cs > aligned_size) cs = aligned_size - off;
            if (cs > 0)
                mbind(aligned_ptr + off, cs, MPOL_BIND, &nodemask,
                      sizeof(nodemask) * 8, flags);
        }
    } else {
        mbind(aligned_ptr, aligned_size, MPOL_BIND, &nodemask,
              sizeof(nodemask) * 8, flags);
    }
}

// Legacy: same body as SCILIB_MV=1. Retained because utils/move_numa.h
// declares it; no active callers in the BLAS wrappers.
void move_numa2(void *ptr, size_t size, int target_node) {
    double tnuma = scilib_second();
    char *aligned_ptr;
    size_t aligned_size;
    page_align(ptr, size, &aligned_ptr, &aligned_size);
    mbind_parallel_chunks(aligned_ptr, aligned_size, target_node,
                          MPOL_MF_MOVE | MPOL_MF_STRICT);
    tnuma = scilib_second() - tnuma;
    DEBUG2(fprintf(stderr, "move_mbind time %15.6f of %lu pages\n",
                   tnuma, aligned_size / page_size_cached()));
}

static int move_numa_call_count = 0;
static size_t move_numa_total_bytes = 0;

void move_numa_print_stats() {
    fprintf(stderr, "    move_numa stats: %d calls, %.1f MB total\n",
            move_numa_call_count, (double)move_numa_total_bytes / 1024.0 / 1024.0);
}

// SCILIB_MV selects the move_numa implementation:
//   0=move_pages, 1=mbind, 2=cudaMemPrefetchAsync, 3=cudaMemAdvise+prefetch,
//   4=no-op, 5=GPU page-touch, 6=VMA-wide mbind on first touch.
// None of 1..6 beat 0 on the real (multi-rank) workload. The switch exists
// for A/B testing — see proxy/README.md and NOTES_HBM_MALLOC.md.

#define VMA_CACHE_SIZE 256
static struct { unsigned long start, end; } vma_cache[VMA_CACHE_SIZE];
static int vma_cache_count = 0;

static int vma_already_done(const void *ptr) {
    unsigned long p = (unsigned long)ptr;
    for (int i = 0; i < vma_cache_count; i++)
        if (p >= vma_cache[i].start && p < vma_cache[i].end) return 1;
    return 0;
}

static int find_vma(const void *ptr, unsigned long *vs, unsigned long *ve) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return 0;
    char line[512];
    unsigned long t = (unsigned long)ptr;
    int found = 0;
    while (fgets(line, sizeof line, f)) {
        unsigned long s, e;
        if (sscanf(line, "%lx-%lx", &s, &e) == 2 && t >= s && t < e) {
            *vs = s; *ve = e; found = 1; break;
        }
    }
    fclose(f);
    return found;
}

static void cache_vma(unsigned long s, unsigned long e) {
    if (vma_cache_count < VMA_CACHE_SIZE) {
        vma_cache[vma_cache_count].start = s;
        vma_cache[vma_cache_count].end = e;
        vma_cache_count++;
    }
}

static int mv_variant(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("SCILIB_MV");
        v = e ? atoi(e) : 0;
    }
    return v;
}

void move_numa(void *ptr, size_t size, int target_node) {
    move_numa_call_count++;
    move_numa_total_bytes += size;
    double tnuma = scilib_second();
    int ps = page_size_cached();

    char *aligned_ptr;
    size_t aligned_size;
    page_align(ptr, size, &aligned_ptr, &aligned_size);
    size_t num_pages = aligned_size / ps;

    int v = mv_variant();
    const char *tag = "move_page ";

    if (v == 0) {
        // move_pages syscall (state-of-the-art baseline on the real workload)
        size_t num_pages_plus = num_pages + 1;
        int *status = malloc(num_pages_plus * sizeof(int));
        int *nodes  = malloc(num_pages_plus * sizeof(int));
        void **page_addrs = malloc(num_pages_plus * sizeof(void *));
        char *char_ptr = ptr;
        #pragma omp parallel for
        for (size_t i = 0; i < num_pages; i++) {
            page_addrs[i] = char_ptr + i * ps;
            nodes[i] = target_node; status[i] = -1;
        }
        page_addrs[num_pages] = char_ptr + size - 1;
        nodes[num_pages] = target_node; status[num_pages] = -1;
        int rc = move_pages(0, num_pages_plus, page_addrs, nodes, status, MPOL_MF_MOVE);
        if (rc < 0) { fprintf(stderr, "move_pages failed\n"); exit(-1); }
        free(page_addrs); free(nodes); free(status);
    } else if (v == 1) {
        mbind_parallel_chunks(aligned_ptr, aligned_size, target_node,
                              MPOL_MF_MOVE | MPOL_MF_STRICT);
        tag = "move_mbind";
    } else if (v == 2 || v == 3) {
        int dev = 0;
        cudaGetDevice(&dev);
        if (v == 3)
            cudaMemAdvise(aligned_ptr, aligned_size,
                          cudaMemAdviseSetPreferredLocation, dev);
        cudaMemPrefetchAsync(aligned_ptr, aligned_size, dev, scilib_cuda_stream);
        cudaStreamSynchronize(scilib_cuda_stream);
        tag = (v == 2) ? "move_pref " : "move_adpf ";
    } else if (v == 4) {
        tag = "move_none ";
    } else if (v == 5) {
        gpu_migrate(aligned_ptr, aligned_size, scilib_cuda_stream);
        tag = "move_gput ";
    } else if (v == 6) {
        // VMA-wide mbind on first touch. Subsequent allocations in the same
        // VMA inherit MPOL_BIND so faults land on the target node without an
        // explicit migration.
        if (vma_already_done(ptr)) {
            tag = "vma cache ";
        } else {
            unsigned long vs, ve;
            unsigned long nodemask = 1UL << target_node;
            if (find_vma(ptr, &vs, &ve)) {
                mbind((void*)vs, ve - vs, MPOL_BIND, &nodemask,
                      sizeof(nodemask) * 8, MPOL_MF_MOVE);
                cache_vma(vs, ve);
                DEBUG2(fprintf(stderr, "  vma-mbind %lx-%lx (%.1f MB)\n",
                               vs, ve, (ve - vs) / 1024.0 / 1024.0));
                tag = "vma mbind ";
            } else {
                mbind(aligned_ptr, aligned_size, MPOL_BIND, &nodemask,
                      sizeof(nodemask) * 8, MPOL_MF_MOVE);
                tag = "fb mbind  ";
            }
        }
    }

    add_to_migrate_cache(ptr, size);
    tnuma = scilib_second() - tnuma;
    DEBUG2(fprintf(stderr, "%s time %15.6f of %lu pages\n", tag, tnuma, num_pages));
}

// Migrate two regions, skipping any already on target node
void move_numa_pair(void *ptr1, size_t size1, void *ptr2, size_t size2, int target_node) {
    if (!is_migrated(ptr1, size1))
        move_numa(ptr1, size1, target_node);
    if (!is_migrated(ptr2, size2))
        move_numa(ptr2, size2, target_node);
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
        "prterun",
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



#include <errno.h>
#include <dlfcn.h>


// ----- HBM-aware malloc interposer -----------------------------------------
// Allocations >= threshold MB get mbind'd to the HBM NUMA node right after
// allocation, so the BLAS wrappers never have to migrate them. Below the
// threshold, allocations pass through unchanged so CPU-side small objects
// keep their DRAM locality.
//
// Toggle via env SCILIB_HBM_MALLOC_MB:
//     unset (default) : on, 64 MB threshold (~17% faster on MuST/LSMS)
//     0               : off (still interposed but no mbind)
//     N>0             : on, N MB threshold

#define HBM_MAL_DEFAULT_MB  64
#define HBM_BOOT_BUF_SIZE   65536
static size_t hbm_mal_thresh = (size_t)HBM_MAL_DEFAULT_MB * 1024 * 1024;
static int    hbm_mal_inited = 0;
static void *(*real_malloc_fn)(size_t) = NULL;
static void *(*real_calloc_fn)(size_t, size_t) = NULL;
static void *(*real_realloc_fn)(void*, size_t) = NULL;
static void  (*real_free_fn)(void*) = NULL;
static __thread int hbm_mal_in_dlsym = 0;
static char  hbm_boot_buf[HBM_BOOT_BUF_SIZE];
static size_t hbm_boot_off = 0;

static void hbm_mal_init(void) {
    if (hbm_mal_inited) return;
    hbm_mal_in_dlsym = 1;
    real_malloc_fn  = dlsym(RTLD_NEXT, "malloc");
    real_calloc_fn  = dlsym(RTLD_NEXT, "calloc");
    real_realloc_fn = dlsym(RTLD_NEXT, "realloc");
    real_free_fn    = dlsym(RTLD_NEXT, "free");
    hbm_mal_in_dlsym = 0;
    const char *e = getenv("SCILIB_HBM_MALLOC_MB");
    if (e) {
        long v = atol(e);
        if (v >= 0) hbm_mal_thresh = (size_t)v * 1024 * 1024;
    }
    hbm_mal_inited = 1;
}

static void hbm_bind_range(void *p, size_t size) {
    char *aligned_ptr;
    size_t aligned_size;
    page_align(p, size, &aligned_ptr, &aligned_size);
    unsigned long nm = 1UL << scilib_hbm_numa;
    mbind(aligned_ptr, aligned_size, MPOL_BIND, &nm, sizeof(nm) * 8, MPOL_MF_MOVE);
}

static int hbm_is_boot(const void *p) {
    return p >= (void*)hbm_boot_buf && p < (void*)(hbm_boot_buf + sizeof(hbm_boot_buf));
}

void *malloc(size_t size) {
    if (hbm_mal_in_dlsym) {
        size_t a = (size + 15) & ~(size_t)15;
        if (hbm_boot_off + a > sizeof(hbm_boot_buf)) return NULL;
        void *p = hbm_boot_buf + hbm_boot_off;
        hbm_boot_off += a;
        return p;
    }
    if (!hbm_mal_inited) hbm_mal_init();
    void *p = real_malloc_fn(size);
    if (p && hbm_mal_thresh && size >= hbm_mal_thresh) hbm_bind_range(p, size);
    return p;
}

void *calloc(size_t n, size_t size) {
    if (hbm_mal_in_dlsym) {
        size_t total = n * size;
        size_t a = (total + 15) & ~(size_t)15;
        if (hbm_boot_off + a > sizeof(hbm_boot_buf)) return NULL;
        void *p = hbm_boot_buf + hbm_boot_off;
        hbm_boot_off += a;
        memset(p, 0, total);
        return p;
    }
    if (!hbm_mal_inited) hbm_mal_init();
    void *p = real_calloc_fn(n, size);
    size_t total = n * size;
    if (p && hbm_mal_thresh && total >= hbm_mal_thresh) hbm_bind_range(p, total);
    return p;
}

void *realloc(void *ptr, size_t size) {
    if (!hbm_mal_inited) hbm_mal_init();
    if (hbm_is_boot(ptr)) {
        // bootstrap chunks aren't tracked — just give a fresh real alloc
        void *p = real_malloc_fn(size);
        if (p && hbm_mal_thresh && size >= hbm_mal_thresh) hbm_bind_range(p, size);
        return p;
    }
    void *p = real_realloc_fn(ptr, size);
    if (p && hbm_mal_thresh && size >= hbm_mal_thresh) hbm_bind_range(p, size);
    return p;
}

void free(void *ptr) {
    if (!ptr) return;
    if (hbm_is_boot(ptr)) return;          // leak the tiny bootstrap chunks
    if (!hbm_mal_inited) hbm_mal_init();
    real_free_fn(ptr);
}

