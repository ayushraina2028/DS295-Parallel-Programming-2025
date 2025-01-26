#include <iostream>
using namespace std;

__global__ void kernelCall(void) {

}

int main() {
    // Hello World Program with GPU Code

    kernelCall<<<1,1>>> (); // Here Kernel Does nothing
    cout << "Hello World" << endl;

    return 0;
}