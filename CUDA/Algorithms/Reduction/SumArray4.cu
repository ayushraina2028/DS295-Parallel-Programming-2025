#include <bits/stdc++.h>

#define threadsPerBlock 16
using namespace std;
using namespace chrono;

/*
Reduction 4: First Add during Load

In Reduction 3, we saw fixed shared memory bank conflict as well as divergent branching. But if we look carefully we will observe that
before first syncthreads the threads are not doing much work, they are only loading the data from global memory to shared memory. We can do little
optimization here. We can directly add the data and then load it to shared memory. This requires half blocks than the previous one.
*/

__global__ void Reduce3(int* device_input, int* device_output) {

    __shared__ int s_data[threadsPerBlock];
    int tid = threadIdx.x;
    int gid = blockIdx.x * (2*blockDim.x) + threadIdx.x;

    s_data[tid] = device_input[gid] + device_input[gid + blockDim.x];
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
    int N = 1 << 28;
    size_t bytes = N*sizeof(int);

    int* host_input = new int[N];
    int* host_output = (int*) malloc((N/(threadsPerBlock*2))*sizeof(int));

    // Initialize the input array to 1
    for(int i = 0;i < N; i++) {
        host_input[i] = 1;
    }

    // Pointers for GPU
    int* device_input; cudaMalloc((void**)&device_input,bytes); cudaMemcpy(device_input, host_input, bytes, cudaMemcpyHostToDevice);
    int* device_output; cudaMalloc((void**)&device_output,(N/(threadsPerBlock*2))*sizeof(int));

    // Each thread Handles 2 Elements
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid /= 2;
    cout << "Blocks Per Grid: " << blocksPerGrid << endl; cout << "Threads Per Block: " << threadsPerBlock << endl;

    // Kernel Call to Reduce the Array

    auto start = getTime();

    Reduce3<<<blocksPerGrid,threadsPerBlock>>>(device_input,device_output);
    cudaDeviceSynchronize();

    int current_N = blocksPerGrid;

    while(current_N >= threadsPerBlock) {

        int newBlocksPerGrid = (current_N + threadsPerBlock - 1) / threadsPerBlock;
        newBlocksPerGrid /= 2;
        cout << "Entered While Loop!, Next Blocks Per Grid, threadsPerBlock: " << newBlocksPerGrid << " " << threadsPerBlock << endl;
        
        Reduce3<<<newBlocksPerGrid,threadsPerBlock>>>(device_output, device_output);
        cudaDeviceSynchronize();
        current_N = newBlocksPerGrid;

    }
    
    // one additional case
    if(current_N > 0 and current_N < threadsPerBlock) {
        int newBlocksPerGrid = 1;
        cout << "Entered While Loop!, Next Blocks Per Grid: " << newBlocksPerGrid << endl;
        
        int newThreadsPerBlock = current_N/2;
        Reduce3<<<newBlocksPerGrid,newThreadsPerBlock>>>(device_output, device_output);
        cudaDeviceSynchronize();
    }

    auto stop = getTime();
    milliseconds duration = duration_cast<milliseconds>(stop - start);
    cout << "Time Taken: " << duration.count() << " ms" << endl;

    cudaMemcpy(host_output,device_output,(N/(threadsPerBlock*2))*sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(device_input); cudaFree(device_output);

    cout << "Array Sum: " << host_output[0] << endl;
    int sumCPU = 0; for(int i = 0;i < N; i++) sumCPU += host_input[i]; 
    
    if(sumCPU == host_output[0]) cout << "CPU and GPU Sums Match!" << endl;
    else cout << "CPU and GPU Sums Do Not Match!" << endl;

    // cout << "Host Output Array: "; for(int i = 0;i < N/threadsPerBlock; i++) cout << host_output[i] << " "; cout << endl;

    delete[] host_input; free(host_output);
    return 0;
}