#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
    
    MPI_Init(&argc, &argv); MPI_Comm comm = MPI_COMM_WORLD;

    int rank; MPI_Comm_rank(comm,&rank);
    int size; MPI_Comm_size(comm,&size);

    double start_time = MPI_Wtime();

    // Simulating some computation
    double wait_time = rank * 0.5; cout << "Process " << rank << " sleeping for " << wait_time << " seconds " << endl;
    sleep(wait_time);

    double end_time = MPI_Wtime();

    // Execution time
    double execution_time = end_time - start_time; cout << "Execution Time: " << execution_time << " for process " << rank << endl;

    MPI_Finalize();
    return 0;
}