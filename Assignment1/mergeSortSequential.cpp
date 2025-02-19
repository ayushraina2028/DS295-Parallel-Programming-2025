#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

void merge(vector<int>& arr, int left, int mid, int right) {
    vector<int> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;
    
    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }
    
    while (i <= mid)
        temp[k++] = arr[i++];
    
    while (j <= right)
        temp[k++] = arr[j++];
    
    for (i = 0; i < k; i++)
        arr[left + i] = temp[i];
}

void mergeSortSequential(vector<int>& arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortSequential(arr, left, mid);
        mergeSortSequential(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

int main() {
    int n = 268435456;
    
    vector<int> arr(n);
    cout << "Enter the elements:\n";
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;
    
    auto start = high_resolution_clock::now();
    mergeSortSequential(arr, 0, n - 1);
    auto stop = high_resolution_clock::now();
    
    // cout << "Sorted elements:\n";
    // for (int i = 0; i < n; i++)
    //     cout << arr[i] << " ";
    // cout << "\n";

    if(is_sorted(arr.begin(), arr.end()))
        cout << "Array is sorted\n";
    else
        cout << "Array is not sorted\n";
    
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Time taken: " << duration.count() << " milliseconds\n";
    
    return 0;
}
