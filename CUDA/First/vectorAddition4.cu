#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

__global__ void VectorAdditionUsing3DBlocks(int* A, int* B, int* Answer, int nX, int nY, int nZ) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if(x < nX and y < nY and z < nZ) {
        int index = x + y*nX + z*nX*nY;
        if(index < nX * nY * nZ) {
            Answer[index] = A[index] + B[index];
        }
    }

}

__global__ void VectorAdditionUsing1DBlock(int* A, int* B, int* C, int N) {
    
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < N) {
        C[index] = A[index] + B[index];
    }

}

high_resolution_clock::time_point getTime() {
    return high_resolution_clock::now();
}

int main() {
    int N = 1000000; // 10^6
    size_t size = N * sizeof(int);

    int *A = (int*) malloc(size), *B = (int*) malloc(size), *C = (int*) malloc(size);

    for(int i = 0;i < N; i++) {
        A[i] = rand() % 15; B[i] = rand() % 20;
    }

    int *dA, *dB, *dC;
    cudaMalloc(&dA,size); cudaMalloc(&dB, size); cudaMalloc(&dC, size);

    cudaMemcpy(dA,A,size,cudaMemcpyHostToDevice); cudaMemcpy(dB,B,size,cudaMemcpyHostToDevice);

    dim3 blockSize(16,8,8);

    int numBlocksInX = (100 + 16 - 1) / 16, numBlocksinY = (100 + 8 - 1) / 8, numBlocksInZ = (100 + 8 - 1) / 8;
    dim3 numBlocks(numBlocksInX,numBlocksinY,numBlocksInZ);

    // Kernel with 3D Blocks
    auto s = getTime();
    VectorAdditionUsing3DBlocks<<<numBlocks,blockSize>>> (dA,dB,dC,100,100,100);
    cudaDeviceSynchronize();
    auto e = getTime();

    nanoseconds duration = duration_cast<nanoseconds> (e - s);
    cout << "Time taken to do Vector Addition using 3D block is: " << duration.count() << endl;

    // Kernel with 1D Blocks
    s = getTime();
    VectorAdditionUsing1DBlock<<<(N + 128 - 1) / 128,128>>>(dA,dB,dC,N);
    cudaDeviceSynchronize();
    e = getTime();

    duration = duration_cast<nanoseconds>(e - s);
    cout << "Time Taken to do Vector Addition using 1D Blocks is: " << duration.count() << endl;

    cudaMemcpy(C,dC,size,cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    free(A); free(B); free(C);
    return 0;
    
}