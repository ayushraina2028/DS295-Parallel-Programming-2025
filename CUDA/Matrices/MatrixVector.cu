#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

// GPU Kernel
__global__ void matrixVectorMult(float* A, float* V,float* Answer, int M, int N) {
    
    int curr_row, curr_col;
    float sum = 0;
    curr_row = blockIdx.x * blockDim.x + threadIdx.x;

    if(curr_row < M) {

        for(curr_col = 0; curr_col < N; curr_col++) {
            sum += A[N*curr_row + curr_col]*V[curr_col];
        }
        Answer[curr_row] = sum;

    }
}

// CPU Code
void matrixVectorMultiplicationCPU(float* A, float* V, float* Answer, int M, int N) {
    for(int i = 0;i < M; i++) {
        Answer[i] = 0;
        for(int j = 0;j < N; j++) {
            Answer[i] += A[i*N + j] * V[j];
        }
    }
}

// Function to measure execution time
high_resolution_clock::time_point getTime() {
    return high_resolution_clock::now();
}

// display functions
void displayMatrix(float* A, int M, int N) {

    cout << "Printing Matrix A: " << endl;
    for(int i = 0;i < M; i++) {
        for(int j = 0;j < N; j++) {
            cout << A[i*M + j] << " ";
        }
        cout << endl;
    }

    return;
}

void displayVector(float* V, int N) {
    cout << "Printing Vector V: " << endl;
    for(int i = 0;i < N; i++) {
        cout << V[i] << " ";
    }
    cout << endl;

    return;
}

int main() {

    // Change these values to 1024, 1024 to see time difference in Cuda vs CPU
    int m = 10240;
    int n = 512;

    size_t size_matrix = m*n*sizeof(float), size_vector = n*sizeof(float), size_answer = m*sizeof(float);

    // Allocating Space on CPU
    float* A = (float*) malloc(size_matrix), *V = (float*) malloc(size_vector), *Answer = (float*) malloc(size_answer);

    // Allocating Space on GPU
    float* dA; cudaMalloc((void**) &dA, size_matrix);
    float* dV; cudaMalloc((void**) &dV,size_vector);
    float* dAnswer; cudaMalloc((void**) &dAnswer,size_answer);

    // Initialization of Matrix
    for(int i = 0;i < m; i++) {
        for(int j = 0;j < n; j++) {
            A[i*n + j] = rand() % 10;
        }
    }

    // Initialization of Vector
    for(int i = 0;i < n; i++) V[i] = rand() % 10;

    // Copy
    cudaMemcpy(dA,A,size_matrix,cudaMemcpyHostToDevice); cudaMemcpy(dV,V,size_vector,cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int numBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;
    
    // Invoking kernel;
    auto start = getTime();
    matrixVectorMult<<<numBlocks, threadsPerBlock>>> (dA,dV,dAnswer,m,n);
    cudaDeviceSynchronize(); // Wait until CUDA kernel finishes completely.
    auto end = getTime();

    // Time Taken By CUDA
    nanoseconds duration = duration_cast<nanoseconds>(end-start);
    cout << "Time Taken by GPU Kernel: " << duration.count() << endl;

    // Copy Result Back
    cudaMemcpy(Answer,dAnswer,size_answer,cudaMemcpyDeviceToHost);

    // Free
    cudaFree(dA); cudaFree(dV); cudaFree(dAnswer);

    // Check Output
    // displayVector(Answer,m);

    // Invoke CPU Call
    start = getTime();
    matrixVectorMultiplicationCPU(A,V,Answer,m,n);
    end = getTime();

    // Time Taken by CPU
    duration = duration_cast<nanoseconds> (end - start);
    cout << "Time Taken by CPU Call: " << duration.count() << endl;

    // displayVector(Answer,m);

    free(A); free(V); free(Answer);

    return 0;
}   

