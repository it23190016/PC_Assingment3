# Matrix Multiplication Parallel Computing Assignment

## Prerequisites
- MSYS2 MinGW with GCC and OpenMP support
- MPICH or OpenMPI (via MSYS2)
- NVIDIA CUDA Toolkit (for CUDA implementation)
- Windows environment with MSYS2

```
PC_Assignment_03/
├── Serial/
├── OpenMP/
├── MPI/
├── CUDA/
└── README.md
```

## Compilation Instructions

### Serial Implementation
```bash
cd Serial
gcc -o matrix_serial matrix_serial.c
```

### OpenMP Implementation
```bash
cd OpenMP
gcc -fopenmp -o matrix_openmp matrix_openmp.c
```

### MPI Implementation
```bash
cd MPI
mpicc -o matrix_mpi matrix_mpi.c
```

### CUDA Implementation
```bash
cd CUDA
nvcc -o matrix_cuda matrix_cuda.cu
```

## Execution Instructions

### Serial
```bash
./matrix_serial
```

### OpenMP
```bash
# Run with different thread counts (MSYS2 MinGW)
export OMP_NUM_THREADS=1 && ./matrix_openmp
export OMP_NUM_THREADS=2 && ./matrix_openmp
export OMP_NUM_THREADS=4 && ./matrix_openmp
export OMP_NUM_THREADS=8 && ./matrix_openmp
```

### MPI
```bash
# Run with different process counts (MSYS2 MinGW)
mpiexec -n 1 ./matrix_mpi
mpiexec -n 2 ./matrix_mpi
mpiexec -n 4 ./matrix_mpi
mpiexec -n 8 ./matrix_mpi
```

### CUDA
```bash
# Modify block size in source code before compilation
./matrix_cuda
```

## Notes
- Matrix size N is defined in source code (default: 1000)
- For CUDA: Use N=5 for testing, N=1000 may cause memory issues
- All implementations use dynamic memory allocation
- Performance results will vary based on hardware specifications

## Hardware Requirements
- Minimum 8GB RAM for N=1000 matrix operations
- NVIDIA GPU with CUDA capability for CUDA implementation
- Multi-core processor recommended for OpenMP/MPI testing