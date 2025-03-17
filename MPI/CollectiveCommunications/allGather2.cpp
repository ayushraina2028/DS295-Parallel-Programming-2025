#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

#define root 0
#define N (1 << 14)

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int nLocal = (N + size - 1) / size; // Ensuring balanced workload distribution
    int startRow = rank * nLocal;
    int endRow = min(startRow + nLocal, N);
    int actualRows = endRow - startRow;

    vector<int> A, x(N, 1), b(N, 0);
    vector<int> LocalArray(actualRows * N, 0), LocalB(actualRows, 0);

    if (rank == root) {
        A.resize(N * N);
        for (int i = 0; i < N * N; i++) A[i] = i + 1; // Initialize matrix A

        // Send only required parts of A to each process
        for (int i = 1; i < size; i++) {
            int start = i * nLocal;
            int end = min(start + nLocal, N);
            int rows = end - start;
            MPI_Send(&A[start * N], rows * N, MPI_INT, i, 0, comm);
        }

        // Copy root's portion of A
        copy(A.begin(), A.begin() + actualRows * N, LocalArray.begin());
    } else {
        MPI_Recv(LocalArray.data(), actualRows * N, MPI_INT, root, 0, comm, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(x.data(), N, MPI_INT, root, comm);

    double start = MPI_Wtime();

    for (int i = 0; i < actualRows; i++) {
        for (int j = 0; j < N; j ++) { // Unrolling for better performance
            LocalB[i] += LocalArray[i * N + j] * x[j];
        }
    }

    vector<int> recvCounts(size), displacements(size);
    for (int i = 0; i < size; i++) {
        recvCounts[i] = min(nLocal, max(0, N - i * nLocal));
        displacements[i] = i * nLocal;
    }

    MPI_Gatherv(LocalB.data(), actualRows, MPI_INT, b.data(), recvCounts.data(), displacements.data(), MPI_INT, root, comm);

    double end = MPI_Wtime();
    double executionTime = end - start;
    cout << "Process " << rank << " Execution Time: " << executionTime * 1e3 << " milliseconds" << endl;

    if (rank == root) {

        double sequentialStart = MPI_Wtime();

        vector<int> correctB(N, 0);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                correctB[i] += A[i * N + j] * x[j];
            }
        }

        double sequentialEnd = MPI_Wtime();
        double sequentialTime = sequentialEnd - sequentialStart;


        bool correct = true;
        for (int i = 0; i < N; i++) {
            if (b[i] != correctB[i]) {
                correct = false;
                break;
            }
        }

        if (correct) cout << "Results are correct!" << endl;
        else cout << "Results are incorrect!" << endl;

        cout << "Execution Time: " << executionTime * 1e3 << " milliseconds" << endl;
        cout << "Sequential Execution Time: " << sequentialTime * 1e3 << " milliseconds" << endl;
        cout << "Speedup: " << sequentialTime / executionTime << endl;
        cout << "Efficiency: " << (sequentialTime / executionTime) / size << endl;
    }

    MPI_Finalize();
    return 0;
}
