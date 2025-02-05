#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

__global__ void MatrixVectorMultUsingSharedMemory(int* A, int* V, int* Answer, int N, int M) {
    
    __shared__ int sharedMemory[512];
    int tid = threadIdx.x;

    sharedMemory[tid] = V[tid];
    __syncthreads();

    int curr_row = blockDim.x * blockIdx.x + tid;
    int sum = 0;

    for(int curr_col = 0; curr_col < M; curr_col++) {
        sum += A[curr_row * M + curr_col] * sharedMemory[curr_col];
    }

    Answer[curr_row] = sum;

}

void matrixvectCPU(int* A, int* V, int* Answer, int N, int M) {

    for(int i = 0;i < N; i++) {
        for(int j = 0; j < M; j++) {
            Answer[i] += A[i*M + j] * V[j];
        }
    }

}

high_resolution_clock::time_point getTime() {
    return high_resolution_clock::now();
}

void displayMatrix(int* A, int N, int M) {
    for(int i = 0;i < N; i++) {
        for(int j = 0; j < M; j++) {
            cout << A[i*M + j] << " ";
        }
        cout << endl;
    }

    return;
}

void displayVector(int* V, int M) {
    for(int i = 0;i < M; i++) {
        cout << V[i] << " ";
    }

    cout << endl;
    return;
}

bool check(int* AnswerCPU, int* AnswerGPU, int M) {

    bool correct = true;

    for(int j = 0;j < M; j++) {

        if(AnswerCPU[j] != AnswerGPU[j]) {
            cout << "Answer not correct" << endl;
            correct = false;
            break;
        }

    }

    return correct;

}

int main() {
    int N = 10240, M = 512;
    size_t sizeMatrix = N*M*sizeof(int), sizeVector = M*sizeof(int), sizeAnswer = N*sizeof(int);

    int* A = (int*) malloc(sizeMatrix);
    for(int i = 0;i < N*M; i++) A[i] = rand() % 15;

    int* V = (int*) malloc(sizeVector);
    for(int i = 0;i < M; i++) V[i] = rand() % 20;

    int AnswerGPU[N] = {0};
    int AnswerCPU[N] = {0};

    int* dA; cudaMalloc((void**)&dA,sizeMatrix); cudaMemcpy(dA,A,sizeMatrix,cudaMemcpyHostToDevice);
    int* dV; cudaMalloc((void**)&dV,sizeVector); cudaMemcpy(dV,V,sizeVector,cudaMemcpyHostToDevice);
    int* dAnswer; cudaMalloc((void**)&dAnswer,sizeAnswer); 

    // Kernel Call
    int threadsPerBlock = M;
    int blocksPerGrid = (N + threadsPerBlock - 1) / (threadsPerBlock);

    auto start = getTime();
    MatrixVectorMultUsingSharedMemory<<<blocksPerGrid, threadsPerBlock>>> (dA,dV,dAnswer,N,M);
    cudaDeviceSynchronize();
    auto end = getTime();

    nanoseconds duration = duration_cast<nanoseconds> (end - start);
    cout << "Time Taken by CUDA Kernel for Matrix Vector MULT: " << duration.count() << endl;

    cudaMemcpy(AnswerGPU,dAnswer,sizeAnswer,cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dV); cudaFree(dAnswer);

    // cout << "Matrix A: " << endl;
    // displayMatrix(A,N,M);

    // cout << "Vector V: " << endl;
    // displayVector(V,M);

    // cout << "Answer GPU: " << endl;
    // displayVector(AnswerGPU,N);

    matrixvectCPU(A,V,AnswerCPU,N,M);

    // cout << "Answer CPU: " << endl;
    // displayVector(AnswerCPU,N);

    bool correct = check(AnswerCPU,AnswerGPU,M);
    if(correct) {
        cout << "Answer is correct" << endl;
    }
    else cout << "Answer is wrong" << endl;

    return 0;
}