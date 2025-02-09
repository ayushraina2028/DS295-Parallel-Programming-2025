#include <bits/stdc++.h>

#define threadsPerBlock 256

using namespace std;
using namespace chrono;

/*
Reduction - 3: Sequential Addressing

It fixes both the Shared Memory Bank Conflicts and Divergent Branching Problems.
Again Loop is a problem because checking loop conditions ar expensive.

*/

__global__ void Reduce2(int* device_input, int* device_output, int N) {

    __shared__ int s_data[threadsPerBlock];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    s_data[tid] = (gid < N) ? device_input[gid] : 0;
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if(tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0) {
        device_output[blockIdx.x] = s_data[0];
    }

}

high_resolution_clock::time_point getTime() {
    return high_resolution_clock::now();
}   

int main() {
    int N = 1 << 16; 
    size_t bytes = N * sizeof(int);

    int* host_input = new int[N];
    int* host_output = (int*)malloc((N / (threadsPerBlock)) * sizeof(int));

    // Initialize input array with 1
    for (int i = 0; i < N; i++) {
        host_input[i] = 1;
    }

    // Allocate GPU memory
    int* device_input, *device_output;
    cudaMalloc((void**)&device_input, bytes);
    cudaMalloc((void**)&device_output, (N / (threadsPerBlock)) * sizeof(int));

    cudaMemcpy(device_input, host_input, bytes, cudaMemcpyHostToDevice);

    // Compute number of blocks
    int blocksPerGrid = (N + (threadsPerBlock  - 1)) / (threadsPerBlock);

    cout << "Blocks Per Grid: " << blocksPerGrid << endl;
    cout << "Threads Per Block: " << threadsPerBlock << endl;

    auto start = getTime();

    // First reduction step
    Reduce2<<<blocksPerGrid, threadsPerBlock>>>(device_input, device_output, N);
    cudaDeviceSynchronize();

    int current_N = blocksPerGrid;  // Update N for next stage

    while (current_N > 1) {
        int new_blocksPerGrid = (current_N + (threadsPerBlock - 1)) / (threadsPerBlock);
        
        cout << "New Blocks Per Grid: " << new_blocksPerGrid << endl;
        
        Reduce2<<<new_blocksPerGrid, threadsPerBlock>>>(device_output, device_output, current_N);
        cudaDeviceSynchronize();

        current_N = new_blocksPerGrid;  // Reduce the problem size
    }

    auto stop = getTime();
    milliseconds duration = duration_cast<milliseconds>(stop - start);
    cout << "Time Taken: " << duration.count() << " ms" << endl;

    cudaMemcpy(host_output, device_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(device_input);
    cudaFree(device_output);

    cout << "Array Sum: " << host_output[0] << endl;

    // Verify with CPU
    long long sumCPU = 0;
    for (int i = 0; i < N; i++) {
        sumCPU += host_input[i];
    }

    if (sumCPU == host_output[0]) {
        cout << "CPU and GPU Sums Match!" << endl;
    } else {
        cout << "CPU and GPU Sums Do Not Match!" << endl;
    }

    delete[] host_input;
    free(host_output);

    return 0;
}
