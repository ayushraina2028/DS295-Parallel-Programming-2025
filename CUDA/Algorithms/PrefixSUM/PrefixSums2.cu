#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

// O(n log n)

__global__ void PrefixSum2(int* A, int* C, int N) {
    int index = threadIdx.x;
    for(int stride = 1; stride < N; stride *= 2) {

        if(index + stride < N) {
            C[index + stride] += C[index];
            // printf("Added A[%d] to C[%d] \n", index, index+stride);
        }

    }
}

high_resolution_clock::time_point getTime() {
    return high_resolution_clock::now();
}
int main() {
    int N = 15;
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
    PrefixSum2<<<blocksPerGrid, threadsPerBlock>>> (dA,dPreSUM,N);

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