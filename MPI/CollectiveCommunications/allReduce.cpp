#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_value = rank + 1;  // Each process has a unique value (1, 2, 3, 4 for 4 processes)
    int global_sum = 0;

    MPI_Allreduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::cout << "Process " << rank << " received sum: " << global_sum << std::endl;

    MPI_Finalize();
    return 0;
}
