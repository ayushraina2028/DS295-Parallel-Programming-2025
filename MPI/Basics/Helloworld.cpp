#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv); 

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get Process Rank */
    MPI_Comm_size(MPI_COMM_WORLD, &size); /* Get Total Number of Process */

    cout << "Hello from process: " << rank << "/" << size << endl;

    MPI_Finalize();
    return 0;   
}