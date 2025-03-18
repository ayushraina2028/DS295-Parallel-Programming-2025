#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

#define MY_INFINITY numeric_limits<int>::max() /* From One.cpp */

class Graph {

private:
    int numberOfVertices;
    vector<vector<int>> AdjacencyMatrix;
    vector<int> LocalVertices;
    vector<int> Level;
    int MPIRank;
    int MPISize;
    int SourceVertex;

public:
    Graph(int Vertices, int SourceVertex) {
        MPI_Comm_rank(MPI_COMM_WORLD, &MPIRank);
        MPI_Comm_size(MPI_COMM_WORLD, &MPISize);

        numberOfVertices = Vertices;
        this->SourceVertex = SourceVertex;
        Level.resize(numberOfVertices,-1);

        /* For Testing assign Local vertices to each rank */
        for(int i = MPIRank; i < numberOfVertices; i += MPISize) {
            LocalVertices.push_back(i);
        }
    }

    void printInfo() {
        cout << "Process: " << MPIRank << " out of " << MPISize << " processes" << endl;   
        cout << "Local Vertices: ";

        for(int vertex : LocalVertices) {
            cout << vertex << " ";
        }
        cout << endl;
    }

};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    Graph graph(10,0); /* 10 vertices, source vertex 0 */
    graph.printInfo();

    MPI_Finalize();
    return 0;
}
