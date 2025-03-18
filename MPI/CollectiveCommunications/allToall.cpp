#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendbuf[4];  // Each process has an array of 4 elements
    int recvbuf[4];  // Buffer to receive 4 elements from different processes

    // Initialize send buffer with unique values for each process
    for (int i = 0; i < 4; i++)
        sendbuf[i] = rank * 10 + i;  // Example: P0 -> {0,1,2,3}, P1 -> {10,11,12,13}, etc.

    // Perform MPI_Alltoall communication
    MPI_Alltoall(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);

    // Print received data
    std::cout << "Process " << rank << " received: ";
    for (int i = 0; i < 4; i++)
        std::cout << recvbuf[i] << " ";
    std::cout << std::endl;

    MPI_Finalize();
    return 0;
}
