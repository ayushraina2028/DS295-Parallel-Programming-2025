#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(rank == 0) {
        int data = 40;
        MPI_Send(&data,1,MPI_INT,1,0,MPI_COMM_WORLD); 
        cout << "Process 0 sent: " << data << " to Process 1" << endl;
    }    
    else if(rank == 1) {
        int receivedData;
        MPI_Status status;
        MPI_Recv(&receivedData,1,MPI_INT,0,0,MPI_COMM_WORLD,&status); 
        cout << "Process 1 received: " << receivedData << " from Process 0" << endl;
    }
    
    MPI_Finalize();
    return 0;
}

/* MPI_Send(&data, count, datatype, destination, tag, communicator); tag - message identifier
        and can be any integer */
/* MPI_Recv(&data, count, datatype, source, tag, communicator, &status); status returns info 
about received message*/