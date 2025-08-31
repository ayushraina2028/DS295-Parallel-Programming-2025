#include <bits/stdc++.h>

using namespace std;
using namespace chrono;


#define CUDA_CHECK_ERROR(call) {  \
    cudaError_t ERROR = call; \
    if(ERROR != cudaSuccess) { \
        cerr << "CUDA ERROR: " << cudaGetErrorString(ERROR) << " in file " << __FILE__ << " at line " << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
    else { \
        cout << "CUDA CALL SUCCESSFULL: " << #call << " in file " << __FILE__ << " at line " << __LINE__ << endl; \
    } \
}  \

// naive version of matrix multiplication on CUDA, we will implement optimized version soon
__global__ void MatrixMult(int* A, int* B, int* C, int N, int K, int M) {
    int curr_row = blockIdx.y * blockDim.y + threadIdx.y;
    int curr_col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    if(curr_row < N and curr_col < M) {
        for(int i = 0;i < K; i++) {
            sum += A[curr_row * K + i] * B[i * M + curr_col];
        }
        C[curr_row * M + curr_col] = sum;
    }
}

void MatrixMultOnCPU(int* A, int* B, int* C, int N, int K, int M) {
    for(int i = 0;i < N; i++) {
        for(int j = 0; j < M; j++) {
            int sum = 0;
            
            for(int k = 0;k < K; k++) {
                sum += A[i*K + k] * B[k*M + j];
            }
            C[i*M + j] = sum;

        }
    }

    return;
}

void fill(int* M, int N) {
    for(int i = 0;i < N; i++) M[i] = rand() % 20;
    return;
}

void display(int* A, int rows, int cols) {
    for(int i = 0;i < rows; i++) {
        for(int j = 0;j < cols; j++) {
            cout << A[i*cols + j] << " ";
        }
        cout << endl;
    }
}

high_resolution_clock::time_point getTime() {
    return high_resolution_clock::now();
}

int main() {
    int N = 1000;
    int K = 1000;
    int M = 1000;

    size_t sizeA = N*K*sizeof(int);
    size_t sizeB = K*M*sizeof(int);
    size_t sizeC = N*M*sizeof(int);

    int *A = (int*)malloc(sizeA), *B = (int*)malloc(sizeB), *C = (int*)malloc(sizeC);

    fill(A,N*K);
    fill(B,K*M);

    int *dA, *dB, *dC;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&dA,sizeA)); CUDA_CHECK_ERROR(cudaMalloc((void**)&dB,sizeB)); CUDA_CHECK_ERROR(cudaMalloc((void**)&dC,sizeC));
    CUDA_CHECK_ERROR(cudaMemcpy(dA,A,sizeA,cudaMemcpyHostToDevice)); CUDA_CHECK_ERROR(cudaMemcpy(dB,B,sizeB,cudaMemcpyHostToDevice));

    // Invoking the kernel
    dim3 threadsPerBlock(2,2,1);

    int blocksInX = (M + threadsPerBlock.x - 1) / threadsPerBlock.x;
    int blocksInY = (N + threadsPerBlock.y - 1) / threadsPerBlock.y;
    dim3 gridSize(blocksInX,blocksInY,1);

    auto s = getTime(); 
    MatrixMult<<<gridSize,threadsPerBlock>>> (dA,dB,dC,N,K,M); CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    auto e = getTime();
    

    milliseconds durationGPU = duration_cast<milliseconds> (e - s);
    cout << "Time taken for CUDA Kernel: " << durationGPU.count() << " milliseconds" << endl;

    CUDA_CHECK_ERROR(cudaMemcpy(C,dC,sizeC,cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(dA)); CUDA_CHECK_ERROR(cudaFree(dB)); CUDA_CHECK_ERROR(cudaFree(dC));

    // int* C_CPU = (int*)malloc(sizeC);

    // s = getTime();
    // MatrixMultOnCPU(A,B,C_CPU,N,K,M);
    // e = getTime();

    // nanoseconds durationCPU = duration_cast<nanoseconds> (e - s);
    // cout << "Time taken for CPU: " << durationCPU.count() << endl;

    // cout << "Matrix A: " << endl; display(A,N,K);
    // cout << "Matrix B: " << endl; display(B,K,M);
    
    // cout << "Matrix C from GPU: " << endl; display(C,N,M);
    // cout << "Matrix C from CPU: " << endl; display(C_CPU,N,M);

    // cout << "SpeedUP: " << (float) (durationCPU.count()) / (durationGPU.count() * 1.0) << endl; // Around 47 times faster

    free(A); free(B); free(C);
    return 0;
}