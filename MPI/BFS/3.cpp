#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

#define myInfinity numeric_limits<int>::max()

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
    Graph(int numVertices, int SourceVertex, int MPIRank, int MPISize) {
        
        numberOfVertices = numVertices;
        this->SourceVertex = SourceVertex;
        Level.resize(numberOfVertices,myInfinity);
        AdjacencyMatrix.resize(numberOfVertices, vector<int> (numberOfVertices,0));
        this->MPIRank = MPIRank;
        this->MPISize = MPISize;

        // Set Level of Source Vertex = 0
        if(MPIRank == 0) {
            Level[SourceVertex] = 0;
        }

        // 1D Partitioning
        int verticesPerProcessor = numberOfVertices / MPISize;
        int remainingVertices = numberOfVertices % MPISize;
        int startingVertex = MPIRank * verticesPerProcessor + min(MPIRank, remainingVertices);
        int numLocalVertices = verticesPerProcessor + (MPIRank < remainingVertices ? 1 : 0);

        // Assigning Local Vertices
        LocalVertices.resize(numLocalVertices);
        for(int i = 0;i < numLocalVertices; i++) {
            LocalVertices[i] = startingVertex + i;
        }

        if(MPIRank == 0) {
            cout << "Graph Initialized with " << numberOfVertices << " vertices, Source Vertex: " << SourceVertex << endl;
        }

        // Print Local Vertices
        cout << "Process: " << MPIRank << " owns vertices: ";
        for(int vertex : LocalVertices) {
            cout << vertex << " ";
        }
        cout << endl;
    }

    // Function to add an Edge to Graph
    void addEdge(int Source, int Destination) {

        if(Source >= 0 && Source < numberOfVertices && Destination >= 0 && Destination < numberOfVertices) {
            AdjacencyMatrix[Source][Destination] = 1;
            AdjacencyMatrix[Destination][Source] = 1;
        }
        else {
            if(MPIRank == 0) {
                cout << "Warning: Invalid Edge (" << Source << ", " << Destination << ") Ignored." << endl;
            }
        }

    }

    // Function to allow manual addition of Edges
    void defineGraph(vector<pair<int, int>> Edges) {
        if(MPIRank == 0) {

            for(auto edge : Edges) {
                addEdge(edge.first, edge.second);
            }
            cout << "Added " << Edges.size() << " Edges to the Graph" << endl;

        }

        // Broadcast the adjacency matrix to all processors
        for(int i = 0;i < numberOfVertices; i++) {
            MPI_Bcast(AdjacencyMatrix[i].data(), numberOfVertices, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    // For Debugging Purposes, Print the Adjacency Matrix
    void printAdjacencyMatrix() {

        if(MPIRank == 0) {
            cout << "Adjacency Matrix: " << endl;
            for(int i = 0;i < numberOfVertices; i++) {
                for(int j = 0;j < numberOfVertices; j++) {

                    cout << AdjacencyMatrix[i][j] << " ";

                }
                cout << endl;
            }
            cout << endl;   
        }
    }

    // Distributed Breadth First Expansion with 1D Partitioning
    void DistributedBreadthFirstSearch() {
        int CurrentLevel = 0;
        bool GlobalDone = false;    

        if(MPIRank == 0) {
            cout << "Starting BFS from Source Vertex: " << SourceVertex << endl;
        }

        while(!GlobalDone) {

            // Find Vertices in Current Frontier
            vector<int> CurrentFrontier;
            for(int localVertex : LocalVertices) {

                if(Level[localVertex] == CurrentLevel) {
                    CurrentFrontier.push_back(localVertex);
                    cout << "Vertex: " << localVertex << " added to Current Frontier" << endl;
                }

            }   

            // Check if all Frontiers are Empty
            int LocalFrontierSize = CurrentFrontier.size();
            int GlobalFrontierSize = 0;

            MPI_Allreduce(&LocalFrontierSize, &GlobalFrontierSize, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            if(GlobalFrontierSize == 0) {
                GlobalDone = true;
            }

            if(GlobalDone) {
                cout << "All Frontiers are Empty. BFS Completed." << endl;
                break;
            }

            // Find Neighbours of Vertices in Current Frontier
            vector<vector<int>> NeighboursOfFrontierVertices(MPISize);
            for(int vertex : CurrentFrontier) {
                for(int neighbour = 0; neighbour < numberOfVertices; neighbour++) {

                    if(AdjacencyMatrix[vertex][neighbour] == 1 && Level[neighbour] == myInfinity) {

                        // Determine which processor owns this vertex
                        int verticesPerProcessor = numberOfVertices / MPISize;
                        int remainingVertices = numberOfVertices % MPISize;
                        int owner = 0;

                        for(int processor = 0; processor < MPISize; processor++) {
                            int startingVertex = processor * verticesPerProcessor + min(processor, remainingVertices);
                            int numLocalVertices = verticesPerProcessor + (processor < remainingVertices ? 1 : 0);

                            // Check if neighbour is owned by this processor by checking if it lies in the range of local vertices
                            if(neighbour >= startingVertex && neighbour < startingVertex + numLocalVertices) {
                                owner = processor;
                                break;
                            }

                        }

                        NeighboursOfFrontierVertices[owner].push_back(neighbour);
                    }
                }
            }

            vector<int> SendCounts(MPISize);
            for(int i = 0;i < MPISize; i++) {
                SendCounts[i] = NeighboursOfFrontierVertices[i].size();
            }

            vector<int> ReceiveCounts(MPISize);
            MPI_Alltoall(SendCounts.data(),1,MPI_INT,ReceiveCounts.data(),1,MPI_INT,MPI_COMM_WORLD);

            vector<int> SendDisplacements(MPISize);
            vector<int> ReceiveDisplacements(MPISize);

            SendDisplacements[0] = 0;
            ReceiveDisplacements[0] = 0;

            for(int i = 1;i < MPISize; i++) {
                SendDisplacements[i] = SendDisplacements[i-1] + SendCounts[i-1];
                ReceiveDisplacements[i] = ReceiveDisplacements[i-1] + ReceiveCounts[i-1];
            }

            int TotalSendCount = SendDisplacements.back() + SendCounts.back();
            int TotalReceiveCount = ReceiveDisplacements.back() + ReceiveCounts.back();

            vector<int> SendBuffer(TotalSendCount);
            for(int i = 0;i < MPISize; i++) {
                copy(NeighboursOfFrontierVertices[i].begin(), NeighboursOfFrontierVertices[i].end(), SendBuffer.begin() + SendDisplacements[i]);
            }

            vector<int> ReceiveBuffer(TotalReceiveCount);
            MPI_Alltoallv(SendBuffer.data(), SendCounts.data(), SendDisplacements.data(), MPI_INT, ReceiveBuffer.data(), ReceiveCounts.data(), ReceiveDisplacements.data(), MPI_INT, MPI_COMM_WORLD);

            // Process Received Neighbours
            for(int i = 0;i < TotalReceiveCount; i++) {

                int Vertex = ReceiveBuffer[i];
                // Check if this vertex is local && not already visited
                for(int localVertex : LocalVertices) {
                    if(localVertex == Vertex && Level[Vertex] == myInfinity) {
                        Level[Vertex] = CurrentLevel + 1;
                        cout << "Vertex: " << Vertex << " marked with Level: " << Level[Vertex] << endl;
                    }
                }
            }

            // Sync all processors
            MPI_Allreduce(MPI_IN_PLACE,Level.data(),numberOfVertices,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            // Move to next level
            CurrentLevel += 1;
        }
    }

    void printResults() {
        if(MPIRank == 0) {
            cout << "BFS Results: " << endl;
            for(int i = 0;i < numberOfVertices; i++) {
                if(Level[i] == myInfinity) {
                    cout << "Vertex " << i << ": Unreachable" << endl;
                }
                else {
                    cout << "Vertex " << i << ": Level " << Level[i] << endl;
                }
            }
        }
    }

};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Graph graph(6,0,rank,size); /* 10 vertices, source vertex 0 */

    vector<pair<int, int>> Edges = {
            {0,1},{1,2},{1,5},{2,5},{2,3},{3,5},{3,4},{5,4}
        };
        graph.defineGraph(Edges);

    graph.printAdjacencyMatrix();

    // Synchronize the graph structure
    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();
    graph.DistributedBreadthFirstSearch();
    double end = MPI_Wtime();

    if(rank == 0) {
        cout << "BFS completed in " << (end - start) << " seconds" << endl;
    }
    graph.printResults();

    MPI_Finalize();
    return 0;
}
