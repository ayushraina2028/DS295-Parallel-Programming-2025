#include <bits/stdc++.h>
using namespace std;

__global__ void VectorAdditionUsingParallelThreads(int* A, int* B, int* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
    int N = 512;
    size_t size = N*sizeof(int);

    int* a = (int*) malloc(size);
    int* b = (int*) malloc(size);
    int* c = (int*) malloc(size);

    int* dA;
    cudaMalloc((void**)&dA,size);

    int* dB;
    cudaMalloc((void**)&dB,size);

    int* dC;
    cudaMalloc((void**)&dC,size);

    for(int i = 0;i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    cudaMemcpy(dA,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,b,size,cudaMemcpyHostToDevice);

    int threadsPerBlock = N;
    int numBlocks = 1;

    VectorAdditionUsingParallelThreads<<<numBlocks,threadsPerBlock>>> (dA,dB,dC);

    cudaMemcpy(c,dC,size,cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    cout << "vector addition using parallel threads" << endl;
    for(int i = 0;i < 10; i++) {
        cout << a[i] << " + " << b[i] << " = " << c[i] << endl;
    }

    free(a);
    free(b);
    free(c);

    return 0;
}