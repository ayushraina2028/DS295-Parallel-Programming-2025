#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

// Cuda Kernel
__global__ void VectorAdditionUsingCUDA(float* A, float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
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
    int N = 100000;
    size_t size = N*sizeof(float);

    // Allocate Space to Input Vectors
    float* hA = (float*) malloc(size);
    float* hB = (float*) malloc(size);
    float* hC = (float*) malloc(size);

    // Initializing Input Vectors
    for(int i = 0;i < N; i++) {
        hA[i] = 2;
        hB[i] = 4;
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
    int threadsPerBlock = 1024;
    int blocksPerGrid = (int)ceil((float)(N/(threadsPerBlock*1.0)));

    // Invoke kernel
    high_resolution_clock::time_point start = high_resolution_clock::now();
    VectorAdditionUsingCUDA<<<blocksPerGrid,threadsPerBlock>>> (dA,dB,dC,N);
    high_resolution_clock::time_point end = high_resolution_clock::now();

    // Measure Execution Time
    nanoseconds duration = duration_cast<nanoseconds> (end-start);
    cout << "Time Taken by CUDA: " << duration.count() << endl;

    //Copy result from device to host again
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

    // Free Host Memory
    free(hA);
    free(hB);
    free(hC);

    return 0;
}