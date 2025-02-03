#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

// CUDA Kernel
/* 
Parallel Programming is easy as long as you don't care about performance, In this version I will spawn n threads to calculate each partial prefix sum
I know, I am going to implement this shit code but still for the purpose of learning, I will do

O(N^2)

 */

__global__ void NaivePrefixSum(int* A, int* C) {
    int index = threadIdx.x;

    int sum = 0;
    for(int i = 0;i <= index; i++) {
        sum += A[i];
    }

    C[index] = sum;
}

high_resolution_clock::time_point getTime() {
    return high_resolution_clock::now();
}

int main() {
    int N = 10;
    size_t size = N*sizeof(int);
    int* A = (int*) malloc(size);
    int* PreSUM = (int*) malloc(size);

    // Initialize both to same data
    for(int i = 0;i < N; i++) {
        A[i] = i;
        PreSUM[i] = i;
    }

    // CUDA Allocation + Copy
    int* dA; cudaMalloc((void**) &dA, size); cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    int* dPreSUM; cudaMalloc((void**) &dPreSUM, size); cudaMemcpy(dPreSUM, PreSUM, size, cudaMemcpyHostToDevice); 

    int threadsPerBlock = N;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Calling 
    NaivePrefixSum<<<blocksPerGrid, threadsPerBlock>>> (dA,dPreSUM);

    // Copy Data Back
    cudaMemcpy(PreSUM, dPreSUM, size, cudaMemcpyDeviceToHost); cudaFree(dA); cudaFree(dPreSUM);

    cout << "Original Array: ";
    for(int i = 0;i < N; i++) cout << A[i] << " ";
    cout << endl;

    cout << "Prefix Sum: ";
    for(int i = 0;i < N; i++) cout << PreSUM[i] << " ";
    cout << endl;

    free(A); free(PreSUM);

    return 0;
}