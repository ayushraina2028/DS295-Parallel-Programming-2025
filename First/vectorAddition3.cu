#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

// Cuda Kernel
__global__ void VectorAdditionUsingBlocksAndThreads(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // This is because every block has same number of threads and it is possible that some threads need not to do anything
    if(i < N) {
        C[i] = A[i] + B[i];
    }

    return;
}

// Cpu Code
void VectorAdditionUsingCPU(float* A, float* B, float* C, int N) {
    for(int i = 0;i < N; i++) {
        C[i] = A[i] + B[i];
    }

    return;
}

int main() {
    int N = 10000000;
    size_t size = N*sizeof(float);

    // Allocate Space to Input Vectors
    float* hA = (float*) malloc(size);
    float* hB = (float*) malloc(size);
    float* hC = (float*) malloc(size);

    // Initializing Input Vectors
    for(int i = 0;i < N; i++) {
        hA[i] = rand() % 100;
        hB[i] = rand() % 100;
    }

    // Allocate Space in Device Memory 
    float* dA;
    cudaMalloc(&dA,size);

    float* dB;
    cudaMalloc(&dB,size);

    float* dC;
    cudaMalloc(&dC,size);

    // copy vectors from host memory to device memory
    cudaMemcpy(dA,hA,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,size,cudaMemcpyHostToDevice);

    // Kernel Parameters
    int threadsPerBlock = 512;
    int blocksPerGrid = (int)ceil((float)(N/(threadsPerBlock*1.0))); // This can also be achieved using (N + M - 1)/M;

    // Invoke kernel, it has to void return type
    high_resolution_clock::time_point start = high_resolution_clock::now();
    VectorAdditionUsingBlocksAndThreads<<<blocksPerGrid,threadsPerBlock>>> (dA,dB,dC,N);
    cudaDeviceSynchronize();
    high_resolution_clock::time_point end = high_resolution_clock::now();

    // Measure Execution Time
    nanoseconds duration = duration_cast<nanoseconds> (end-start);
    cout << "Time Taken by CUDA: " << duration.count() << endl;

    //Copy result from device to host again (This is only way to copy from Device Memory)
    cudaMemcpy(hC,dC,size,cudaMemcpyDeviceToHost);

    // Free Device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Call CPU Code
    start = high_resolution_clock::now();
    VectorAdditionUsingCPU(hA,hB,hC,N);
    end = high_resolution_clock::now();

    // Measure CPU Execution Time
    duration = duration_cast<nanoseconds> (end - start);
    cout << "Time Taken by CPU: " << duration.count() << endl;

    cout << "vector addition using multiple threads and blocks" << endl;
    for(int i = 0;i < 10; i++) {
        cout << hA[i] << " + " << hB[i] << " = " << hC[i] << endl;
    }

    // Free Host Memory
    free(hA);
    free(hB);
    free(hC);

    return 0;
}

/*
__global__ -> This is execution space specifier and <<< >>> is execution configuration syntax
gridSize = Number of blocks inside the grid
blockSize = Number of threads inside the grid
call to __global__ is asynchronous means, it will not wait until the call is completed.To Get synchronisation, we need to use sync_threads();

Hence gridSize * blockSize number of threads are launched and each thread calls the kernel

2. __device__ -> This is another execution space specifier which declares that a function is executed and is callable
from device only

3. __host__ -> executed and callable from host only, By default any function is host function

For a particular function __global__ and __device__ cannot be used together, __global__ and __host__ cannot be used together
but __device__ and __host__ can be used together;

Summary
__device__ is executed on device and is callable only from device
__global__ is executed on device and is callable from device/host
__host__ is executed on host and is callable only from host

*/