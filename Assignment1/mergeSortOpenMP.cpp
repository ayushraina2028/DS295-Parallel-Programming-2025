#include <omp.h>
#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

// Merge function to combine two sorted arrays
void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    
    while (i <= mid and j <= right) {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }
    
    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    for (i = 0; i < k; i++) {
        arr[left + i] = temp[i];
    }

}

// Recursive merge sort function with OpenMP parallelization
void mergeSortParallel(vector<int>& arr, int left, int right, int depth = 0) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        
        // Parallelize only up to a certain depth to avoid overhead
        if (depth < 4) {
            #pragma omp parallel sections
            {
                #pragma omp section
                mergeSortParallel(arr, left, mid, depth + 1);
                
                #pragma omp section
                mergeSortParallel(arr, mid + 1, right, depth + 1);
            }
        } else {
            mergeSortParallel(arr, left, mid, depth + 1);
            mergeSortParallel(arr, mid + 1, right, depth + 1);
        }
        
        merge(arr, left, mid, right);
    }
}

// Sequential merge sort for comparison
void mergeSortSequential(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortSequential(arr, left, mid);
        mergeSortSequential(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

int main() {
    // Set the number of threads to use
    omp_set_num_threads(24);
    
    // Generate test data
    const int SIZE = 268435456; // 10 million elements
    vector<int> arr_parallel(SIZE);
    vector<int> arr_sequential(SIZE);
    
    // Initialize with random numbers
    srand(time(0));
    for (int i = 0; i < SIZE; i++) {
        arr_parallel[i] = rand();
        arr_sequential[i] = arr_parallel[i];
    }
    
    // Time parallel version
    double start_time = omp_get_wtime();
    mergeSortParallel(arr_parallel, 0, SIZE - 1);
    double parallel_time = omp_get_wtime() - start_time;
    
    // Time sequential version
    start_time = omp_get_wtime();
    mergeSortSequential(arr_sequential, 0, SIZE - 1);
    double sequential_time = omp_get_wtime() - start_time;
    
    // Verify results
    bool sorted_correctly = is_sorted(arr_parallel.begin(), arr_parallel.end());
    
    // Print results
    cout << "Array size: " << SIZE << endl;
    cout << "Parallel execution time: " << parallel_time << " milliseconds" << endl;
    cout << "Sequential execution time: " << sequential_time << " seconds" << endl;
    cout << "Speedup: " << 60 / (parallel_time*(int)pow(10,3)) << "x" << endl;
    cout << "Sorted correctly: " << (sorted_correctly ? "Yes" : "No") << endl;
    
    return 0;
}