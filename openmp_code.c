#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000
#define SERIAL_TIME 5.512

int main() {
    int i, j, k;
    double **A, **B, **C;
    double start, end;

    // Allocate memory for matrices
    A = (double**)malloc(N * sizeof(double*));
    B = (double**)malloc(N * sizeof(double*));
    C = (double**)malloc(N * sizeof(double*));
    for (i = 0; i < N; i++) {
        A[i] = (double*)malloc(N * sizeof(double));
        B[i] = (double*)malloc(N * sizeof(double));
        C[i] = (double*)malloc(N * sizeof(double));
    }

    // Initialize input matrices
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = i + j;
            B[i][j] = i - j;
            C[i][j] = 0.0;
        }
    }

    start = omp_get_wtime();

    // Matrix multiplication (same logic as serial, parallelized)
    #pragma omp parallel for private(j, k)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    end = omp_get_wtime();

    double execution_time = end - start;
    double speedup = SERIAL_TIME / execution_time;
    printf("OpenMP execution:\nThreads used: %d\nTime: %.6f sec\nSpeedup: %.2fx\n\n", omp_get_max_threads(), execution_time, speedup);

    // Free memory
    for (i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}

