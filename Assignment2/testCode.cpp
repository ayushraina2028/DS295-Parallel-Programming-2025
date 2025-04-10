#include <mpi.h>
#include <iostream>
#include <unistd.h> // for gethostname()

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    char hostname[256];

    // Get the rank (process ID) and total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    gethostname(hostname, 256); // Get the machine's hostname

    std::cout << "Hello from rank " << rank << " out of " << size
              << " on " << hostname << std::endl;

    MPI_Finalize();
    return 0;
}
