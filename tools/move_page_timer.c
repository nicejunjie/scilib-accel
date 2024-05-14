

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <numaif.h>
#include <unistd.h>
#include <numa.h>

#define NUM_PAGES 100

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

    // Allocate memory on NUMA node 0 for float array
    void* ptr = numa_alloc_onnode(page_size * NUM_PAGES, 0);
    if (ptr == NULL) {
        perror("numa_alloc_onnode");
        exit(EXIT_FAILURE);
    }

    // Array to store individual page move times
    double page_times[NUM_PAGES];
    double total_time = 0;

    printf("\n\n--------------------------\n\n");
    // Loop to move 10 pages one at a time and measure time for each move
    for (int i = 0; i < NUM_PAGES; ++i) {
        // Set up variables for timing
        struct timespec start, end;

        // Get the start time
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Move a page from NUMA node 0 to node 1
        int status;
        void* page_ptr = (char*)ptr + i * page_size;
        if (move_pages(0, 1, &page_ptr, NULL, &status, MPOL_MF_MOVE) < 0) {
            perror("move_pages");
            exit(EXIT_FAILURE);
        }

        // Get the end time
        clock_gettime(CLOCK_MONOTONIC, &end);

        // Calculate the elapsed time in seconds
        double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        // Store the individual page move time
        page_times[i] = elapsed_time;
        total_time += elapsed_time;

        if (i < 10) {
            double bandwidth_bytes = page_size / elapsed_time;
            double bandwidth_GBps = bandwidth_bytes / (1e9);
            double bandwidth_GiBps = bandwidth_bytes / (1024 * 1024 * 1024);
            printf("  Time(%2d): %.9f seconds, Bandwidth: %10.2f GB/s (%10.2f GiB/s)\n", i+1, elapsed_time, bandwidth_GBps, bandwidth_GiBps);
        }

    }

    // Sort the page_times array
    qsort(page_times, NUM_PAGES, sizeof(double), compare_doubles);

    // Calculate minimum, mean, median, and maximum time taken for each page move
    double min_time = page_times[0];
    double max_time = page_times[NUM_PAGES - 1];
    double mean_time = total_time / NUM_PAGES;
    double median_time = (NUM_PAGES % 2 == 0) ? (page_times[NUM_PAGES / 2 - 1] + page_times[NUM_PAGES / 2]) / 2 : page_times[NUM_PAGES / 2];

    // Calculate the total data transferred in bytes
    size_t total_bytes_transferred = page_size * NUM_PAGES;

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
    printf("\n");
    printf("Page size: %ld bytes\n", page_size);
    printf("Pages moved successfully from NUMA node 0 to node 1\n");
    printf("Time taken:\n");
    printf("  -Minimum: %.9f seconds\n", min_time);
    printf("  -Mean: %.9f seconds\n", mean_time);
    printf("  -Median: %.9f seconds\n", median_time);
    printf("  -Maximum: %.9f seconds\n", max_time);
    printf("Memory bandwidth based on time:\n");
    printf("  -Minimum: %8.2f GB/s (%8.2f GiB/s)\n", min_bandwidth_GBps, min_bandwidth_GiBps);
    printf("  -Mean: %8.2f GB/s (%8.2f GiB/s)\n", mean_bandwidth_GBps, mean_bandwidth_GiBps);
    printf("  -Median: %8.2f GB/s (%8.2f GiB/s)\n", median_bandwidth_GBps, median_bandwidth_GiBps);
    printf("  -Maximum: %8.2f GB/s (%8.2f GiB/s)\n", max_bandwidth_GBps, max_bandwidth_GiBps);
    printf("\n\n--------------------------\n\n");

    // Free the allocated memory
    numa_free(ptr, page_size * NUM_PAGES);

    return 0;
}
