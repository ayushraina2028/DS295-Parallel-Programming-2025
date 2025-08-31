#include <iostream>
#include <vector>
#include <chrono>
#include <random>

// Function to initialize a matrix with random values
void initializeMatrix(std::vector<std::vector<float>>& matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = dist(gen);
        }
    }
}

// Function to perform sequential matrix multiplication
void matrixMultiplication(const std::vector<std::vector<float>>& A, 
                         const std::vector<std::vector<float>>& B, 
                         std::vector<std::vector<float>>& C) {
    int A_rows = A.size();
    int A_cols = A[0].size();
    int B_cols = B[0].size();
    
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            C[i][j] = 0.0f;
            for (int k = 0; k < A_cols; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to print a matrix (for small matrices)
void printMatrix(const std::vector<std::vector<float>>& matrix) {
    for (const auto& row : matrix) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Define matrix dimensions
    int A_rows = 1000;
    int A_cols = 1000;
    int B_rows = A_cols;  // Must match for multiplication
    int B_cols = 1000;
    
    // Initialize matrices
    std::vector<std::vector<float>> A(A_rows, std::vector<float>(A_cols));
    std::vector<std::vector<float>> B(B_rows, std::vector<float>(B_cols));
    std::vector<std::vector<float>> C(A_rows, std::vector<float>(B_cols));
    
    // Fill matrices with random values
    initializeMatrix(A, A_rows, A_cols);
    initializeMatrix(B, B_rows, B_cols);
    
    // Perform matrix multiplication and measure time
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplication(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate and print execution time
    std::chrono::duration<double> duration = end - start;
    std::cout << "Matrix multiplication completed in " << duration.count() << " seconds" << std::endl;
    
    
    return 0;
}