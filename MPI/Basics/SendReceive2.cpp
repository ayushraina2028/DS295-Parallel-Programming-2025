#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank;
    MPI_Comm_rank(comm,&rank);

    vector<int> nums(10,1);

    if(rank == 0) {
        cout << "Process: " << rank << " will process elements from: 0 -> 4" << endl;
        for(int i = 0;i <= 4; i++) nums[i]++;

        MPI_Send(&nums[5],5,MPI_INT,1,0,comm); /* Send Half of the Elements */
        MPI_Status status; MPI_Recv(&nums[5],5,MPI_INT,1,0,comm,&status); /* Receive Processed Elements */
    }

    if(rank == 1) {
        MPI_Status status; MPI_Recv(&nums[5],5,MPI_INT,0,0,comm,&status); /* Receive Elements to Process */
                
        for(int i = 5;i <= 9; i++) nums[i]++;
        MPI_Send(&nums[5],5,MPI_INT,0,0,comm); /* Send Back Processes Elements all at once */
        
    }

    cout << "Work Completed for Process: " << rank << endl;
    cout << "Updated Array: ";
    for(int i = 0;i < 10; i++) cout << nums[i] << " ";
    cout << endl;
    MPI_Finalize();
    
    return 0;
}