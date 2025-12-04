#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <stdlib.h>

#define N 1000
#define SERIAL_TIME 5.512

__global__ void matmul_kernel(double *A, double *B, double *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

int main() {
    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)malloc(N * N * sizeof(double));
    double *dA, *dB, *dC;

    for (int i = 0; i < N * N; i++) {
        A[i] = (i / N) + (i % N);
        B[i] = (i / N) - (i % N);
    }

    cudaMalloc(&dA, N * N * sizeof(double));
    cudaMalloc(&dB, N * N * sizeof(double));
    cudaMalloc(&dC, N * N * sizeof(double));

    cudaMemcpy(dA, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N * N * sizeof(double), cudaMemcpyHostToDevice);

    // Test different thread configurations
    for (int t = 1; t <= 32; t *= 2) {
        int thread_x = t;
        int thread_y = t;
        
        dim3 threads(thread_x, thread_y);
        dim3 blocks((N + thread_x - 1) / thread_x, (N + thread_y - 1) / thread_y);
        
        clock_t start = clock();
        matmul_kernel<<<blocks, threads>>>(dA, dB, dC);
        cudaDeviceSynchronize();
        clock_t end = clock();
        
        cudaMemcpy(C, dC, N * N * sizeof(double), cudaMemcpyDeviceToHost);
        
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        double speedup = SERIAL_TIME / time_taken;
        printf("CUDA execution:\n");
        printf("Threads per block: (%d*%d) = %d\n", thread_x, thread_y, thread_x * thread_y);
        printf("Number of blocks: (%d,%d)\n", blocks.x, blocks.y);
        printf("Time: %.6f sec, Speedup: %.2fx\n\n", time_taken, speedup);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(A);
    free(B);
    free(C);
    return 0;
}
