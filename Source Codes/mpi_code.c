#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 1000
#define SERIAL_TIME 5.512

int main(int argc, char *argv[]) {
    int i, j, k;
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    if (rank == 0) {
        // Initialize input matrices
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                A[i][j] = i + j;
                B[i][j] = i - j;
                C[i][j] = 0.0;
            }
        }
    }

    // Broadcast B matrix to all processes
    for (i = 0; i < N; i++) {
        MPI_Bcast(B[i], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Calculate work distribution
    int rows_per_proc = N / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? N : start_row + rows_per_proc;

    start = MPI_Wtime();

    // Matrix multiplication (same logic as serial)
    for (i = start_row; i < end_row; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    end = MPI_Wtime();

    if (rank == 0) {
        double execution_time = end - start;
        double speedup = SERIAL_TIME / execution_time;
        printf("MPI execution:\nProcesses used: %d\nTime: %.6f sec\nSpeedup: %.2fx\n\n", size, execution_time, speedup);
    }

    // Free memory
    for (i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return 0;
}
