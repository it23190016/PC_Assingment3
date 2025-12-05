#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000  // Matrix size (N x N)

int main() {
    int i, j, k;
    double **A, **B, **C;
    clock_t start_serial, end_serial;

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

    // Measure execution time
    start_serial = clock();

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    end_serial = clock();
    double serial_time = (double)(end_serial - start_serial) / CLOCKS_PER_SEC;

    printf("Serial execution:\n");
    printf("    Threads used       : 1\n");
    printf("    Execution time     : %.6f seconds\n\n", serial_time);

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
