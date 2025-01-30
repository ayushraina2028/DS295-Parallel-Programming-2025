#include <bits/stdc++.h>
using namespace std;

__global__ void VectorAdditionUsingParallelBlocks(int* A, int* B, int* C) {
    int i = blockIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
    int *a, *b, *c;
    int *dA, *dB, *dC;
    int N = 512;
    size_t size = N*sizeof(int);

    // Allocate space in GPU
    cudaMalloc((void**) &dA, size);
    cudaMalloc((void**) &dB, size);
    cudaMalloc((void**) &dC, size);

    a = (int*) malloc(size);
    b = (int*) malloc(size);
    c = (int*) malloc(size);

    // Initialize
    for(int i = 0;i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    // Copy
    cudaMemcpy(dA,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,b,size,cudaMemcpyHostToDevice);

    int threadsPerBlock = 1;
    int numBlocks = N;

    // Invoke the kernel
    VectorAdditionUsingParallelBlocks<<<numBlocks, threadsPerBlock>>> (dA,dB,dC);
    
    // copy Back
    cudaMemcpy(c,dC,size,cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Print first 10
    cout << "vector addition using parallel blocks" << endl;
    for(int i = 0;i < 10; i++) {
        cout << a[i] << " + " << b[i] << " = " << c[i] << endl;
    }

    free(a);
    free(b);
    free(c);

    return 0;
}