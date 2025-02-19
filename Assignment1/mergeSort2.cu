#include <bits/stdc++.h>
using namespace std;
using namespace chrono;

#define BLOCK_SIZE 128

high_resolution_clock::time_point getTime() {
    return high_resolution_clock::now();
}

__global__ void ParallelMergeKernel(int* input, int* output, int size, int window_size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= size) return;

    // Calculate required indices for sublists
    int start_one = (tid * 2) * window_size;
    int end_one = min(start_one + window_size, size);

    int start_two = end_one;
    int end_two = min(start_two + window_size, size);

    if(start_one >= size) return;

    // Initialize pointers for output
    int i = start_one;
    int j = start_two;
    int k = start_one;

    // Binary search based merge
    while(i < end_one && j < end_two) {

        // Find position of input[j] in first subarray
        if(i < end_one) {
            int low = i, high = end_one, position = i;
            
            while(low < high) {
                
                int mid = low + (high - low) / 2;
                if(input[mid] <= input[j]) {
                    low = mid + 1;
                    position = low;
                } else {
                    high = mid;
                }

            }
            
            // Copy elements from first subarray up to the found position
            while(i < position) {
                output[k++] = input[i++];
            }
            
            // Copy element from second subarray
            output[k++] = input[j++];
        }
    }

    // Copy remaining elements from first subarray
    while(i < end_one) {
        output[k++] = input[i++];
    }

    // Copy remaining elements from second subarray
    while(j < end_two) {
        output[k++] = input[j++];
    }
}

void mergeSortInvoker(int* input, int size, vector<milliseconds>& duration) {

    int* device_input; cudaMalloc(&device_input, size * sizeof(int)); cudaMemcpy(device_input, input, size * sizeof(int), cudaMemcpyHostToDevice);
    int* device_output; cudaMalloc(&device_output, size * sizeof(int));


    auto start = getTime();
    // Calling the kernel for Log(n) times
    for(int window_size = 1; window_size < size; window_size *= 2) {
        
        int threadsPerBlock = (size + 2*window_size - 1) / (2*window_size);
        int blocksPerGrid = (threadsPerBlock + BLOCK_SIZE - 1) / BLOCK_SIZE;

        ParallelMergeKernel<<<blocksPerGrid, threadsPerBlock>>>(device_input, device_output, size, window_size);
        cudaDeviceSynchronize();

        // swap the roles of input and output
        int* temp = device_input;
        device_input = device_output;
        device_output = temp;

    }
    auto end = getTime();

    milliseconds time = duration_cast<milliseconds>(end - start);
    duration.push_back(time);
    cout << "Time taken for sorting: " << time.count() << " milliseconds" << endl;

    cudaMemcpy(input, device_input, size * sizeof(int), cudaMemcpyDeviceToHost); cudaFree(device_input); cudaFree(device_output);
}


int main() {

    vector<milliseconds> duration;
    for(int M = 0; M < 5; M++) {
        
        int size = 16777216;
        int* input = new int[size];

        // Initialize the array with random values
        for(int i = 0; i < size; i++) {
            input[i] = rand() % 1000000;
        }


        mergeSortInvoker(input, size, duration);


        if(is_sorted(input, input + size)) {
            cout << "Array is sorted!" << endl;
        } else {
            cout << "Array is not sorted!" << endl;
        }

    }

    // print averge time taken for sorting
    long long sum = 0;
    for(auto time: duration) {
        sum += time.count();
    }
    cout << "Average time taken for sorting: " << sum / duration.size() << " milliseconds" << endl;
    cout << "Speedup: " << 4329.0 / ((sum / duration.size())*1.0) << endl;

    return 0;
}