//gcc -o fftw_test test_fftw3d-omp.c -lfftw3 -lm
//pgcc -o test_fftw test_fftw3d-omp.c -lfftw3_omp -lfftw3 -lm -mp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <fftw3.h>

// ANSI color codes for green and red
#define GREEN "\033[0;32m"
#define RED "\033[0;31m"
#define RESET "\033[0m"

// Function to initialize the array with random values
void initialize_input(fftw_complex *input, int N0, int N1, int N2) {
#pragma omp parallel for
    for (int i = 0; i < N0; ++i) {
        for (int j = 0; j < N1; ++j) {
            for (int k = 0; k < N2; ++k) {
                int index = (i * N1 * N2) + (j * N2) + (k);
                input[index][0] = drand48(); // Real part
                input[index][1] = drand48(); // Imaginary part
            }
        }
    }
}

// Function to validate the output and print the min, mean, and max difference
void validate_and_print_difference(fftw_complex *original, fftw_complex *transformed, int N0, int N1, int N2) {
    double tolerance = 1e-10;
    double min_diff = 1e20;
    double max_diff = -1e20;
    double sum_diff = 0.0;
    int count = 0;

#pragma omp parallel for
    for (int i = 0; i < N0; ++i) {
        for (int j = 0; j < N1; ++j) {
            for (int k = 0; k < N2; ++k) {
                int index = (i * N1 * N2) + (j * N2) + (k);
                double real_diff = fabs(original[index][0] - transformed[index][0]);
                double imag_diff = fabs(original[index][1] - transformed[index][1]);
                double diff = sqrt(real_diff * real_diff + imag_diff * imag_diff);

                if (diff < min_diff) min_diff = diff;
                if (diff > max_diff) max_diff = diff;
                sum_diff += diff;
                count++;
            }
        }
    }

    double mean_diff = sum_diff / count;

    printf("Minimum difference: %e\n", min_diff);
    printf("Mean difference: %e\n", mean_diff);
    printf("Maximum difference: %e\n", max_diff);

    int is_valid = max_diff <= tolerance;
    if (is_valid) {
        printf(GREEN "Validation successful.\n" RESET);
    } else {
        printf(RED "Validation failed.\n" RESET);
    }
}

// Function to measure time
double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <N0> <N1> <N2>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N0 = atoi(argv[1]);
    int N1 = atoi(argv[2]);
    int N2 = atoi(argv[3]);

    fftw_complex *input = fftw_malloc(sizeof(fftw_complex) * N0 * N1 * N2);
    fftw_complex *output = fftw_malloc(sizeof(fftw_complex) * N0 * N1 * N2);
    fftw_complex *transformed_back = fftw_malloc(sizeof(fftw_complex) * N0 * N1 * N2);

    initialize_input(input, N0, N1, N2);

    // Initialize FFTW threads
    fftw_init_threads();
    fftw_plan_with_nthreads(omp_get_max_threads());

    struct timespec start, end;

    printf("========================================\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    fftw_plan forward_plan = fftw_plan_dft_3d(N0, N1, N2, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan backward_plan = fftw_plan_dft_3d(N0, N1, N2, output, transformed_back, FFTW_BACKWARD, FFTW_ESTIMATE);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double setup_time = get_time_diff(start, end);
    printf("Plan FFT time    : %f seconds\n", setup_time);

    // Measure forward FFT execution time
    clock_gettime(CLOCK_MONOTONIC, &start);
    fftw_execute(forward_plan);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double forward_time = get_time_diff(start, end);
    printf("Forward FFT time : %f seconds\n", forward_time);

    // Measure backward FFT execution time
    clock_gettime(CLOCK_MONOTONIC, &start);
    fftw_execute(backward_plan);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double backward_time = get_time_diff(start, end);
    printf("Backward FFT time: %f seconds\n", backward_time);
    printf("----------------------------------------\n");

    // Normalize the transformed back array
#pragma omp parallel for
    for (int i = 0; i < N0 * N1 * N2; ++i) {
        transformed_back[i][0] /= (N0 * N1 * N2);
        transformed_back[i][1] /= (N0 * N1 * N2);
    }

    validate_and_print_difference(input, transformed_back, N0, N1, N2);
    printf("========================================\n");

    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_cleanup_threads();
    fftw_free(input);
    fftw_free(output);
    fftw_free(transformed_back);

    return EXIT_SUCCESS;
}

