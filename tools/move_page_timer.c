//gcc -o move_page move_page_timer.c -lnuma

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <numaif.h>
#include <unistd.h>
#include <numa.h>
#include <omp.h>

#define NUM_PAGES 10000
#define ORIG_NUMA 0
#define DEST_NUMA 1


double mysecond()
{
    struct timespec measure;

    // Get the current time as the start time
    clock_gettime(CLOCK_MONOTONIC, &measure);

    // Return the elapsed time in seconds
    return (double)measure.tv_sec + (double)measure.tv_nsec * 1e-9;
}


void print_system_info() {
    // Print CONFIG_ARM64_64K_PAGES
    char command[256];
    snprintf(command, sizeof(command), "grep CONFIG_ARM64_64K_PAGES /boot/config-$(uname -r)");
    printf("CONFIG_ARM64_64K_PAGES:\n");
    system(command);

    printf("--- OS Setup --\n");
    // Print PAGESIZE
    long pagesize = sysconf(_SC_PAGESIZE);
    printf("PAGESIZE: %ld\n", pagesize);

    // Print Hugepagesize
    FILE *file = fopen("/proc/meminfo", "r");
    if (file != NULL) {
        char line[256];
        while (fgets(line, sizeof(line), file)) {
            if (strstr(line, "Hugepagesize:") != NULL) {
                printf("%s", line);
                break;
            }
        }
        fclose(file);
    } else {
        perror("Failed to open /proc/meminfo");
    }

    // Print transparent_hugepage/defrag
    file = fopen("/sys/kernel/mm/transparent_hugepage/defrag", "r");
    if (file != NULL) {
        char defrag[256];
        if (fgets(defrag, sizeof(defrag), file) != NULL) {
            printf("transparent_hugepage/defrag: %s", defrag);
        }
        fclose(file);
    } else {
        perror("Failed to open /sys/kernel/mm/transparent_hugepage/defrag");
    }

    // Print transparent_hugepage/enabled
    file = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
    if (file != NULL) {
        char enabled[256];
        if (fgets(enabled, sizeof(enabled), file) != NULL) {
            printf("transparent_hugepage/enabled: %s", enabled);
        }
        fclose(file);
    } else {
        perror("Failed to open /sys/kernel/mm/transparent_hugepage/enabled");
    }

    // Print numa_balancing
    file = fopen("/proc/sys/kernel/numa_balancing", "r");
    if (file != NULL) {
        char numa_balancing[256];
        if (fgets(numa_balancing, sizeof(numa_balancing), file) != NULL) {
            printf("numa_balancing: %s", numa_balancing);
        }
        fclose(file);
    } else {
        perror("Failed to open /proc/sys/kernel/numa_balancing");
    }
}




int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

int main() {
    // Get the system's actual page size
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1) {
        perror("sysconf");
        exit(EXIT_FAILURE);
    }

    printf("\n--------------------------------\n\n");
    print_system_info();

    printf("\nNumber of pages: %d\n\n", NUM_PAGES);


    // Allocate memory on NUMA node 0 
    char* ptr = numa_alloc_onnode(page_size * NUM_PAGES, ORIG_NUMA);
    if (ptr == NULL) {
        perror("numa_alloc_onnode");
        exit(EXIT_FAILURE);
    }
    memset(ptr, 0, page_size * NUM_PAGES);

    // Array to store individual page move times
    double page_times[NUM_PAGES];
    double total_time = 0;

    int *status = malloc(NUM_PAGES*sizeof(int));
    int *nodes = malloc(NUM_PAGES*sizeof(int));
    void **page_addrs = malloc(NUM_PAGES * sizeof(void *));

    // Populate the array with page addresses
    for (int i = 0; i < NUM_PAGES; ++i) {
        page_addrs[i] = ptr + i * page_size;
        nodes[i]=DEST_NUMA;
        status[i]=-1;
    }

    double elapsed_time;

    printf("---- Single page migration experiments: ----\n\n");
    // Loop to move 10 pages one at a time and measure time for each move
    for (int i = 0; i < NUM_PAGES; ++i) {
    
        elapsed_time = -mysecond();
        // Move a page from NUMA node 0 to node 1
        if (move_pages(0, 1, &page_addrs[i], &nodes[i], &status[i], MPOL_MF_MOVE) < 0) {
            perror("move_pages");
            exit(EXIT_FAILURE);
        }
        elapsed_time += mysecond();

        // Store the individual page move time
        page_times[i] = elapsed_time;
        total_time += elapsed_time;

        if (i < 10) {
            double bandwidth_bytes = page_size / elapsed_time;
            double bandwidth_GBps = bandwidth_bytes / (1e9);
            double bandwidth_GiBps = bandwidth_bytes / (1024 * 1024 * 1024);
            printf("  Time(%2d): %.9f seconds, Bandwidth: %6.2f GB/s (%6.2f GiB/s)\n", i+1, elapsed_time, bandwidth_GBps, bandwidth_GiBps);
        }

    }

    // Free the allocated memory
    numa_free(ptr, page_size * NUM_PAGES);

    // Sort the page_times array
    qsort(page_times, NUM_PAGES, sizeof(double), compare_doubles);

    // Calculate minimum, mean, median, and maximum time taken for each page move
    double min_time = page_times[0];
    double max_time = page_times[NUM_PAGES - 1];
    double mean_time = total_time / NUM_PAGES;
    double median_time = (NUM_PAGES % 2 == 0) ? (page_times[NUM_PAGES / 2 - 1] + page_times[NUM_PAGES / 2]) / 2 : page_times[NUM_PAGES / 2];


    // Calculate the memory bandwidth in bytes per second for min, mean, median, and max times
    double min_bandwidth_bytes = page_size / max_time;
    double mean_bandwidth_bytes = page_size / mean_time;
    double median_bandwidth_bytes = page_size / median_time;
    double max_bandwidth_bytes = page_size / min_time;

    // Convert bandwidth to gigabytes per second and gibibytes per second
    double min_bandwidth_GBps = min_bandwidth_bytes / (1e9);
    double min_bandwidth_GiBps = min_bandwidth_bytes / (1024 * 1024 * 1024);
    double mean_bandwidth_GBps = mean_bandwidth_bytes / (1e9);
    double mean_bandwidth_GiBps = mean_bandwidth_bytes / (1024 * 1024 * 1024);
    double median_bandwidth_GBps = median_bandwidth_bytes / (1e9);
    double median_bandwidth_GiBps = median_bandwidth_bytes / (1024 * 1024 * 1024);
    double max_bandwidth_GBps = max_bandwidth_bytes / (1e9);
    double max_bandwidth_GiBps = max_bandwidth_bytes / (1024 * 1024 * 1024);

    // Print the results
    printf("\nTime taken:\n");
    printf("  +Minimum:  %.9f seconds\n", min_time);
    printf("  +Mean:     %.9f seconds\n", mean_time);
    printf("  +Median:   %.9f seconds\n", median_time);
    printf("  +Maximum:  %.9f seconds\n", max_time);
    printf("Memory bandwidth based on page migration:\n");
    printf("  +Minimum:  %6.2f GB/s (%6.2f GiB/s)\n", min_bandwidth_GBps, min_bandwidth_GiBps);
    printf("  +Mean:     %6.2f GB/s (%6.2f GiB/s)\n", mean_bandwidth_GBps, mean_bandwidth_GiBps);
    printf("  +Median:   %6.2f GB/s (%6.2f GiB/s)\n", median_bandwidth_GBps, median_bandwidth_GiBps);
    printf("  +Maximum:  %6.2f GB/s (%6.2f GiB/s)\n", max_bandwidth_GBps, max_bandwidth_GiBps);


// bulk migration

    // Allocate memory on NUMA node 0 
    ptr = numa_alloc_onnode(page_size * NUM_PAGES, ORIG_NUMA);
    if (ptr == NULL) {
        perror("numa_alloc_onnode");
        exit(EXIT_FAILURE);
    }
    memset(ptr, 0, page_size * NUM_PAGES);

    int rc=0;
    elapsed_time = -mysecond();
    rc=move_pages(0 /*self memory*/, NUM_PAGES, page_addrs, nodes, status, 0);
    elapsed_time += mysecond();
    if(rc!=0) {
        if(rc > 0) fprintf(stderr, "warning: %d pages not moved\n", rc);
        if(rc < 0) {fprintf(stderr, "error: page migration failed\n"); exit(-1);}
    }

    // Free the allocated memory
    numa_free(ptr, page_size * NUM_PAGES);

    // Calculate the total data transferred in bytes
    size_t total_bytes_transferred = page_size * NUM_PAGES;
    double bulk_time = elapsed_time;
    double bulk_bandwidth_bytes = total_bytes_transferred / bulk_time;
    double bulk_bandwidth_GBps = bulk_bandwidth_bytes / (1e9);
    double bulk_bandwidth_GiBps = bulk_bandwidth_bytes / (1024 * 1024 * 1024);


    printf("\n---- Bulk page migration experiments: ----\n\n");
    printf("Time taken:\n");
    printf("  +Bulk:     %.9f seconds\n", bulk_time);
    printf("Memory bandwidth based on page migration:\n");
    printf("  +Bulk:     %6.2f GB/s (%6.2f GiB/s)\n", bulk_bandwidth_GBps, bulk_bandwidth_GiBps);


// omp parallel bulk migration

    ptr = numa_alloc_onnode(page_size * NUM_PAGES, ORIG_NUMA);
    if (ptr == NULL) {
        perror("numa_alloc_onnode");
        exit(EXIT_FAILURE);
    }
    memset(ptr, 0, page_size * NUM_PAGES);

    elapsed_time = -mysecond();
    #pragma omp parallel
    {
        int thread_rc = 0;
        #pragma omp for
        for (size_t i = 0; i < NUM_PAGES; i += omp_get_num_threads()) {
            size_t start = i;
            size_t end = ((i + omp_get_num_threads()) < NUM_PAGES) ? (i + omp_get_num_threads()) : NUM_PAGES;
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
    elapsed_time += mysecond();
    numa_free(ptr, page_size * NUM_PAGES);

    double pbulk_time = elapsed_time;
    double pbulk_bandwidth_bytes = total_bytes_transferred / pbulk_time;
    double pbulk_bandwidth_GBps = pbulk_bandwidth_bytes / (1e9);
    double pbulk_bandwidth_GiBps = pbulk_bandwidth_bytes / (1024 * 1024 * 1024);


    printf("\n---- OpenMP Bulk page migration experiments: ----\n");
    #pragma omp parallel
    #pragma omp single
    printf("OMP_NUM_THREADS=%d\n\n",omp_get_num_threads());
    printf("Time taken:\n");
    printf("  +OMP Bulk: %.9f seconds\n", pbulk_time);
    printf("Memory bandwidth based on page migration:\n");
    printf("  +OMP Bulk: %6.2f GB/s (%6.2f GiB/s)\n", pbulk_bandwidth_GBps, pbulk_bandwidth_GiBps);



    printf("\n--------------------------------\n\n");

    return 0;
}
