#include <bits/stdc++.h>    

using namespace std;
using namespace chrono;

#define BLOCK_SIZE 128

__global__ void mergeKernel(int* input, int* output, int size, int width) {
    // Each thread handles one merge operation of two width-sized segments
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the starting index for this thread's merge operation
    int start = idx * 2 * width;
    
    // If start is beyond array size, return
    if (start >= size) return;
    
    // Calculate boundaries for first and second segments
    int left1 = start;
    int right1 = min(start + width, size);
    int left2 = right1;
    int right2 = min(left2 + width, size);
    
    int i = left1;
    int j = left2;
    int k = left1;
    
    // Merge the two segments
    while (i < right1 && j < right2) {
        if (input[i] <= input[j]) {
            output[k++] = input[i++];
        } else {
            output[k++] = input[j++];
        }
    }
    
    // Copy remaining elements from first segment
    while (i < right1) {
        output[k++] = input[i++];
    }
    
    // Copy remaining elements from second segment
    while (j < right2) {
        output[k++] = input[j++];
    }
}

void mergeSortGPU(int* input, int size, vector<milliseconds>& duration) {
    int* d_input;
    int* d_output;
    
    // Allocate device memory
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));
    
    // Copy input data to device
    cudaMemcpy(d_input, input, size * sizeof(int), cudaMemcpyHostToDevice);
    
    int* d_ping = d_input;
    int* d_pong = d_output;

    auto start = high_resolution_clock::now();
    
    for (int width = 1; width < size; width *= 2) {
        // Calculate required number of thread blocks
        // We need ceil(size/(2*width)) threads
        int numThreadsNeeded = (size + (2 * width - 1)) / (2 * width);
        int numBlocks = (numThreadsNeeded + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        mergeKernel<<<numBlocks, BLOCK_SIZE>>>(d_ping, d_pong, size, width);
        cudaDeviceSynchronize();
        
        // Swap buffers
        int* temp = d_ping;
        d_ping = d_pong;
        d_pong = temp;
    }

    auto end = high_resolution_clock::now();
    duration.push_back(duration_cast<milliseconds>(end - start));
    cout << "Time taken for sorting: " << duration.back().count() << " milliseconds" << endl;
    
    // If final result is in d_pong, copy it to d_ping
    if (d_ping != d_input) {
        cudaMemcpy(d_input, d_output, size * sizeof(int), cudaMemcpyDeviceToDevice);
        d_ping = d_input;
    }
    
    // Copy result back to host
    cudaMemcpy(input, d_ping, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

// Test function
int main() {
    for(int M = 0; M < 5; M++) {
        
        int size = 16777216;
        int* input = new int[size];

        // Initialize the array with random values
        for(int i = 0; i < size; i++) {
            input[i] = rand() % 1000000;
        }

        vector<milliseconds> duration;
        mergeSortGPU(input, size, duration);

        milliseconds averageTime = accumulate(duration.begin(), duration.end(), milliseconds(0)) / duration.size();
        cout << "Average Time taken for sorting: " << averageTime.count() << " milliseconds" << endl;

        if(is_sorted(input, input + size)) {
            cout << "Array is sorted!" << endl;
        } else {
            cout << "Array is not sorted!" << endl;
        }

    }
}