#include <bits/stdc++.h>
#include <mpi.h> 

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    int dataToSend = rank; /* Each process will send its rank to other processor */
    int dataToReceive[size]; /* Each process will receive rank from other processors */

    MPI_Allgather(&dataToSend,1,MPI_INT,dataToReceive,1,MPI_INT,comm);

    // Printing Gathered Data at each process
    cout << "Process: " << rank << " received: ";
    for(auto R : dataToReceive) cout << R << " ";
    cout << endl;


    MPI_Finalize();
    return 0;
}