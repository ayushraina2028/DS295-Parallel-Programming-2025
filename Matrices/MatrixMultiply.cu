#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

__global__ void MatrixMult(int* A, int* B, int* C, int N, int K, int M) {
    
}

void fill(int* M, int N) {
    for(int i = 0;i < N; i++) M[i] = rand() % 20;
    return;
}
int main() {
    int N = 10;
    int K = 20;
    int M = 30;

    size_t sizeA = N*K*sizeof(int);
    size_t sizeB = K*M*sizeof(int);
    size_t sizeC = N*M*sizeof(int);

    int *A = (int*)malloc(sizeA), *B = (int*)malloc(sizeB), *C = (int*)malloc(sizeC);

    fill(A,N*K);
    fill(B,K*M);

    float *dA, *dB, *dC;
    cudaMalloc((void**)&dA,sizeA); cudaMalloc((void**)&dB,sizeB); cudaMalloc((void**)&dC,sizeC);
    cudaMemcpy(dA,A,sizeA,cudaMemcpyHostToDevice); cudaMemcpy(dB,B,sizeB,cudaMemcpyDeviceToHost);

    // Invoking the kernel

    cudaMemcpy(C,dC,sizeC,cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    for(int i = 0;i < N; i++) {
        for(int j = 0;j < M; j++) {
            cout << C[i*N + j] << " ";
        }
        cout << endl;
    }

    free(A); free(B); free(C);
    return 0;
}