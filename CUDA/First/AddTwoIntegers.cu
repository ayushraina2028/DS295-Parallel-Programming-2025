#include <iostream>
using namespace std;

__global__ void addIntegers(int* a, int* b, int* c) {
    *c = *a + *b;
}

int main() {
    
    int a, b, c;
    int *dA, *dB, *dC;

    size_t size = sizeof(int);

    // Allocate space on GPU
    cudaMalloc(&dA,size);
    cudaMalloc(&dB,size);
    cudaMalloc(&dC,size);

    // Initialize;
    a = 4;
    b = 4;

    // Copy
    cudaMemcpy(dA,&a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,&b,size,cudaMemcpyHostToDevice);

    // Launch kernel
    addIntegers<<<1,1>>> (dA,dB,dC);

    // Copy back to CPU
    cudaMemcpy(&c,dC,size,cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    cout << "Answer is -> " << c << endl;
    return 0;
}