#include <bits/stdc++.h>
using namespace std;

__global__ void EnumerationSort(int* A, int* POS) {
    int threadID = threadIdx.x;
    int blockID = blockIdx.x;

    if(A[threadID] > A[blockID]) {
        atomicAdd(&POS[threadID],1); // This is required so that all threads will add 1 correctly.
    }
}

int main() {
    int N = 100;
    size_t size = N*sizeof(int);

    int* A = (int*)malloc(size);
    int* pos = (int*)malloc(size);

    for(int i = 0;i < N; i++) {
        A[i] = rand() % 100;
        pos[i] = 0;
    }

    int *dA; cudaMalloc((void**)&dA,size); cudaMemcpy(dA,A,size,cudaMemcpyHostToDevice);
    int *dPos; cudaMalloc((void**)&dPos,size); cudaMemcpy(dPos,pos,size,cudaMemcpyHostToDevice);

    int threadsPerBlock = N;
    int blocksPerGrid = N;

    // Kernel Call;
    EnumerationSort<<<blocksPerGrid, threadsPerBlock>>>(dA,dPos);
    cudaMemcpy(pos,dPos,size,cudaMemcpyDeviceToHost); cudaFree(dA); cudaFree(dPos);
    
    int* answer = (int*)malloc(size);
    for(int i = 0;i < N; i++) answer[i] = 0;

    // Fill Elements at correct place
    for(int i = 0;i < N; i++) {
        int idx = pos[i];

        if(answer[idx] != 0) {
            int k = 1;
            bool flag = true;

            while(flag) {
                if(answer[idx+k] == 0) {
                    answer[idx+k] = answer[idx];
                    flag = false;
                }
                else k++;
            }
        }
        else answer[idx] = A[i];
    }

    cout << "Original Array: ";
    for(int i = 0;i < N; i++) cout << A[i] << " ";
    cout << endl;

    cout << "Sorted Array: ";
    for(int i = 0;i < N; i++) {
        cout << answer[i] << " ";
    }
    cout << endl;

    free(A); free(pos); free(answer);
    return 0;
}
