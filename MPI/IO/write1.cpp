#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Open File in write mode */
    MPI_File file;
    MPI_File_open(comm,"output.txt",MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL,&file); /* Creates if file does not exist and opens it for write */

    /* Prepare a message */
    char message[50];
    snprintf(message,sizeof(message),"Hello from processor %d of %d\n",rank,size);

    /* Prepare offset */
    int offset = rank * strlen(message);

    /* Write to file */
    MPI_File_write_at(file,offset,message,strlen(message),MPI_CHAR,MPI_STATUS_IGNORE);

    /* Close file */
    MPI_File_close(&file);

    MPI_Finalize();
    return 0;
}