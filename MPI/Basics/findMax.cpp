#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
    int N = 20;
    int arr[N] = {0}; for(int i = 0;i < N; i++) arr[i] = rand() % 120;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank; MPI_Comm_rank(comm,&rank);
    int size; MPI_Comm_size(comm,&size);

    if(size != 2) {
        cout << "This program works only with 2 processors \n" << endl;
        exit(0);
    }

    int Local_max1 = INT_MIN, Local_max2 = INT_MIN, Global_Max = INT_MIN;
    if(rank == 0) {
        int start = 0;
        int end = N/2;
        cout << "Process: " << rank << " start: " << start << " end: " << end << endl;

        for(int i = start; i < end; i++) {
            Local_max1 = max(Local_max1, arr[i]);
        }

        MPI_Send(&Local_max1,1,MPI_INT,1,0,comm); /* Send Left Max */
    }
    else {
        int start = N/2;
        int end = start + N - N/2;
        cout << "Process: " << rank << " start: " << start << " end: " << end << endl;

        for(int i = start; i < end; i++) {
            Local_max2 = max(Local_max2, arr[i]); 
        }

        int othermax; MPI_Status status;
        MPI_Recv(&othermax,1,MPI_INT,0,0,comm,&status); /* Receive Left Max */

        Global_Max = max(othermax,Local_max2); /* Compute Global Maximum */

        cout << "Array: ";
        for(int i = 0;i < N; i++) cout << arr[i] << " ";
        cout << endl;
        cout << "Maximum Element in the array: " << Global_Max << endl;
    }


    MPI_Finalize();
    return 0;
}