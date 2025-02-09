#include <bits/stdc++.h>
using namespace std;

__device__ int binarySearch(int* nums, int target, int N) {
    int lo = 0;
    int hi = N-1;
    int mid;

    while(lo <= hi) {
        mid = lo + (hi-lo)/2;

        if(nums[mid] == target) return mid;
        else if(nums[mid] < target) lo = mid+1;
        else hi = mid-1;

    }

    return lo;
}

__global__ void merge2SortedLists1(int* A, int* B, int* C, int N, int M) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N+M) return;

    if(idx < N) {
        int otherIdx = binarySearch(B,A[idx],M);
        C[idx+otherIdx] = A[idx];
        printf("Idx = %d, otherIdx = %d for element %d \n", idx, otherIdx, A[idx]);
    }
    else {
        int otherIdx = binarySearch(A,B[idx-N],N);
        C[idx-N+otherIdx] = B[idx-N];
        printf("Idx = %d, otherIdx = %d for element %d \n", idx-N, otherIdx, B[idx-N]);
    }
}

int main() {
    int N = 1;  
    int M = 1;
    size_t sizeA = N*sizeof(int);
    size_t sizeB = M*sizeof(int);

    int *A = (int*)malloc(sizeA), *B = (int*)malloc(sizeB), *C = (int*)malloc(sizeA + sizeB);
    for(int i = 0;i < N; i++) {
        A[i] = 2*i;
        B[i] = 2*i+1;
    }

    int* dA; cudaMalloc((void**)&dA, sizeA); cudaMemcpy(dA,A,sizeA,cudaMemcpyHostToDevice);
    int* dB; cudaMalloc((void**)&dB, sizeB); cudaMemcpy(dB,B,sizeB,cudaMemcpyHostToDevice);
    int* dC; cudaMalloc((void**)&dC, sizeA + sizeB);

    // Kernel Call
    int threadsPerBlock = 32;
    int blocksPerGrid = (N + M + threadsPerBlock - 1) / (threadsPerBlock);

    merge2SortedLists1<<<blocksPerGrid, threadsPerBlock>>> (dA,dB,dC,N,M);

    cudaMemcpy(C,dC,sizeA+sizeB,cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    cout << "Array 1: ";
    for(int i = 0;i < N; i++) cout << A[i] << " ";
    cout << endl;

    cout << "Array 2: ";
    for(int i = 0;i < M; i++) cout << B[i] << " ";
    cout << endl;

    cout << "Sorted Array: ";
    for(int i = 0;i < N+M; i++) cout << C[i] << " ";
    cout << endl;

    free(A); free(B); free(C);
    return 0;
}