#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

__global__ void MatrixVectorMultUsingSharedMemory(int* A, int* V, int* Answer, int N, int M) {
    
    __shared__ int sharedMemory[512];

    int localID = threadIdx.x;
    int curr_row = blockDim.x * blockIdx.x + threadIdx.x;

    sharedMemory[localID] = V[localID];
    __syncthreads();

    int curr_col; int sum = 0;
    if(curr_row < N) {

        for(curr_col = 0; curr_col < M; curr_col++) {
            sum += A[curr_row * M + curr_col] * sharedMemory[curr_col];
        }
        Answer[curr_row] = sum;
    }

}

high_resolution_clock::time_point getTime() {
    return high_resolution_clock::now();
}

int main() {
    int N = 10000, M = 512;
    size_t sizeMatrix = N*M*sizeof(int), sizeVector = M*sizeof(int), sizeAnswer = N*sizeof(int);

    int* A = (int*)malloc(sizeMatrix); int* V = (int*)malloc(sizeVector); int* Answer = (int*)malloc(sizeAnswer);

    // Initializing
    for(int i = 0;i < N*M; i++) A[i] = rand() % 10;
    for(int i = 0;i < M; i++) V[i] = rand() % 10;

    int* dA; cudaMalloc((void**)&dA,sizeMatrix); cudaMemcpy(dA,A,sizeMatrix,cudaMemcpyHostToDevice);
    int* dV; cudaMalloc((void**)&dV,sizeVector); cudaMemcpy(dV,V,sizeVector,cudaMemcpyHostToDevice);
    int* dAnswer; cudaMalloc((void**)&dAnswer,sizeAnswer); 

    // Kernel Call
    int threadsPerBlock = 512;
    int blocksPerGrid = (N + threadsPerBlock - 1) / (threadsPerBlock);

    auto start = getTime();
    MatrixVectorMultUsingSharedMemory<<<blocksPerGrid, threadsPerBlock>>> (dA,dV,dAnswer,N,M);
    cudaDeviceSynchronize();
    auto end = getTime();

    nanoseconds duration = duration_cast<nanoseconds> (end - start);
    cout << "Time Taken by CUDA Kernel for Matrix Vector MULT: " << duration.count() << endl;

    cudaMemcpy(Answer,dAnswer,sizeAnswer,cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dV); cudaFree(dAnswer);

    // cout << "Matrix A: " << endl;
    // for(int i = 0;i < N; i++) {
    //     for(int j = 0;j < M; j++) {
    //         cout << A[i*M + j] << " ";
    //     }
    //     cout << endl;
    // }

    // cout << "Vector V: " << endl;
    // for(int i = 0;i < M; i++) {
    //     cout << V[i] << " ";
    // }
    // cout << endl;

    // cout << "Answer: " << endl;
    // for(int i = 0;i < N; i++) {
    //     cout << Answer[i] << " ";
    // }
    // cout << endl;

    free(A); free(V); free(Answer);
    return 0;
}