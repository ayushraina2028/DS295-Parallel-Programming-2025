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

// GPU Kernel
__global__ void matrixVectorMult(float* A, float* V,float* Answer, int M, int N) {
    
    int curr_row, curr_col;
    double sum;
    curr_row = blockIdx.x * blockDim.x + threadIdx.x;

    if(curr_row < M) {
        for(curr_col = 0; curr_col < N; curr_col++) {
            sum += A[M*curr_row + curr_col]*V[curr_col];
        }
    }

    Answer[curr_row] = sum;


}

// CPU Code
void matrixVectorMultiplicationCPU(float* A, float* V, float* Answer, int M, int N) {
    for(int i = 0;i < M; i++) {
        Answer[i] = 0;
        for(int j = 0;j < N; j++) {
            Answer[i] += A[i*M + j] * V[j];
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
    int m = 10;
    int n = 20;

    size_t size_matrix = m*n*sizeof(float);
    size_t size_vector = n*sizeof(float);
    size_t size_answer = m*sizeof(float);

    // Allocating Space on CPU
    float* A = (float*) malloc(size_matrix);
    float* V = (float*) malloc(size_vector);
    float* Answer = (float*) malloc(size_answer);

    // Allocating Space on GPU
    float* dA;
    CUDA_CHECK_ERROR(cudaMalloc((void**) &dA, size_matrix));

    float* dV;
    CUDA_CHECK_ERROR(cudaMalloc((void**) &dV,size_vector));

    float* dAnswer;
    CUDA_CHECK_ERROR(cudaMalloc((void**) &dAnswer,size_answer));

    // Initialization of Matrix
    for(int i = 0;i < m; i++) {
        for(int j = 0;j < n; j++) {
            A[i*m + j] = rand() % 10;
        }
    }

    // Initialization of Vector
    for(int i = 0;i < n; i++) V[i] = rand() % 10;

    // Display
    // displayMatrix(A,m,n);
    // displayVector(V,n);

    // Copy
    CUDA_CHECK_ERROR(cudaMemcpy(dA,A,size_matrix,cudaMemcpyHostToDevice));    
    CUDA_CHECK_ERROR(cudaMemcpy(dV,V,size_vector,cudaMemcpyHostToDevice));

    int threadsPerBlock = 128;
    int numBlocks = (int)ceil((float)(m/(threadsPerBlock* 1.0)));
    
    // Invoking kernel;
    auto start = getTime();
    matrixVectorMult<<<numBlocks, threadsPerBlock>>> (dA,dV,dAnswer,m,n);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize()); // Wait until CUDA kernel finishes completely.
    auto end = getTime();

    // Time Taken By CUDA
    nanoseconds duration = duration_cast<nanoseconds>(end-start);
    cout << "Time Taken by GPU Kernel: " << duration.count() << endl;

    // Copy Result Back
    CUDA_CHECK_ERROR(cudaMemcpy(Answer,dAnswer,size_answer,cudaMemcpyDeviceToHost));

    // Free
    CUDA_CHECK_ERROR(cudaFree(dA));
    CUDA_CHECK_ERROR(cudaFree(dV));
    CUDA_CHECK_ERROR(cudaFree(dAnswer));

    // Check Output
    displayVector(Answer,m);

    // Invoke CPU Call
    start = getTime();
    matrixVectorMultiplicationCPU(A,V,Answer,m,n);
    end = getTime();

    // Time Taken by CPU
    duration = duration_cast<nanoseconds> (end - start);
    cout << "Time Taken by CPU Call: " << duration.count() << endl;

    displayVector(Answer,m);

    free(A);
    free(V);
    free(Answer);

    return 0;
}   

