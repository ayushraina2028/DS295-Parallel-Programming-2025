#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <limits>
#include <string>
#include <sstream>
#include <mpi.h>

#define INFINITY std::numeric_limits<int>::max()

class Graph {
private:
    int numVertices;              // Total number of vertices
    std::vector<std::vector<int>> adjacencyMatrix;  // Adjacency matrix
    std::vector<int> localVertices;  // Vertices owned by this processor
    std::vector<int> level;       // BFS level for each vertex
    int rank;                     // MPI Rank
    int size;                     // MPI Size
    int sourceVertex;             // Source vertex for BFS

public:
    // Constructor
    Graph(int n, int source, int mpiRank, int mpiSize) : 
        numVertices(n), 
        adjacencyMatrix(n, std::vector<int>(n, 0)),
        level(n, INFINITY),
        rank(mpiRank),
        size(mpiSize),
        sourceVertex(source) {
        
        // Set source vertex level to 0
        if (rank == 0) {
            level[sourceVertex] = 0;
        }
        
        // Calculate vertices owned by this processor using 1D partitioning
        int verticesPerProc = numVertices / size;
        int remainder = numVertices % size;
        
        int startVertex = rank * verticesPerProc + std::min(rank, remainder);
        int numLocal = verticesPerProc + (rank < remainder ? 1 : 0);
        
        // Assign local vertices    
        localVertices.resize(numLocal);
        for (int i = 0; i < numLocal; i++) {
            localVertices[i] = startVertex + i;
        }
        
        if (rank == 0) {
            std::cout << "Graph initialized with " << numVertices << " vertices, source: " << sourceVertex << std::endl;
        }
        
        std::cout << "Rank " << rank << " owns vertices: ";
        for (int v : localVertices) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }
    
    // Add an edge to the graph
    void addEdge(int src, int dest) {
        if (src >= 0 && src < numVertices && dest >= 0 && dest < numVertices) {
            adjacencyMatrix[src][dest] = 1;
            adjacencyMatrix[dest][src] = 1;  // For undirected graph
        } else {
            if (rank == 0) {
                std::cout << "Warning: Invalid edge (" << src << ", " << dest << ") ignored." << std::endl;
            }
        }
    }
    
    // Load graph from file
    bool loadFromFile(const std::string& filename) {
        if (rank == 0) {
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cout << "Error: Could not open file " << filename << std::endl;
                return false;
            }
            
            std::string line;
            while (std::getline(file, line)) {
                // Skip comments and empty lines
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                
                std::istringstream iss(line);
                int src, dest;
                if (iss >> src >> dest) {
                    addEdge(src, dest);
                }
            }
            
            std::cout << "Graph loaded from file: " << filename << std::endl;
        }
        
        // Broadcast the adjacency matrix to all processors
        for (int i = 0; i < numVertices; i++) {
            MPI_Bcast(adjacencyMatrix[i].data(), numVertices, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        return true;
    }
    
    // Allow manual addition of edges
    void defineGraph(const std::vector<std::pair<int, int>>& edges) {
        if (rank == 0) {
            for (const auto& edge : edges) {
                addEdge(edge.first, edge.second);
            }
            std::cout << "Added " << edges.size() << " edges to the graph" << std::endl; 
        }
        
        // Broadcast the adjacency matrix to all processors
        for (int i = 0; i < numVertices; i++) {
            MPI_Bcast(adjacencyMatrix[i].data(), numVertices, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    
    // For interactive addition of edges (rank 0 only)
    void inputEdgesInteractively() {
        if (rank == 0) {
            std::cout << "Enter edges (source destination), one per line. Enter -1 -1 to finish:\n";
            int src, dest;
            while (true) {
                std::cout << "Edge: ";
                std::cin >> src >> dest;
                if (src == -1 && dest == -1) {
                    break;
                }
                addEdge(src, dest);
            }
        }
        
        // Broadcast the adjacency matrix to all processors
        for (int i = 0; i < numVertices; i++) {
            MPI_Bcast(adjacencyMatrix[i].data(), numVertices, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    
    // Print the adjacency matrix (for debugging)
    void printAdjacencyMatrix() {
        if (rank == 0) {
            std::cout << "\nAdjacency Matrix:\n";
            for (int i = 0; i < numVertices; i++) {
                for (int j = 0; j < numVertices; j++) {
                    std::cout << adjacencyMatrix[i][j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
    
    // Perform distributed BFS
    void distributedBFS() {
        int currentLevel = 0;
        bool globalDone = false;
        
        if (rank == 0) {
            std::cout << "Starting BFS from source vertex " << sourceVertex << std::endl;
        }
        
        while (!globalDone) {
            // Find local frontier vertices (vertices with level == currentLevel)
            std::vector<int> frontier;
            for (int localIdx : localVertices) {
                if (level[localIdx] == currentLevel) {
                    frontier.push_back(localIdx);
                }
            }
            
            // Check if all processors have empty frontiers
            int localDone = frontier.empty() ? 1 : 0;
            int allDone;
            MPI_Allreduce(&localDone, &allDone, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
            globalDone = (allDone == 1);
            
            if (globalDone) {
                break;
            }
            
            // Find neighbors of frontier vertices
            std::vector<std::vector<int>> neighborsToSend(size);
            for (int v : frontier) {
                for (int j = 0; j < numVertices; j++) {
                    if (adjacencyMatrix[v][j] && level[j] == INFINITY) {
                        // Determine which processor owns this vertex
                        int verticesPerProc = numVertices / size;
                        int remainder = numVertices % size;
                        int owner = 0;
                        
                        for (int p = 0; p < size; p++) {
                            int start = p * verticesPerProc + std::min(p, remainder);
                            int count = verticesPerProc + (p < remainder ? 1 : 0);
                            
                            if (j >= start && j < start + count) {
                                owner = p;
                                break;
                            }
                        }
                        
                        neighborsToSend[owner].push_back(j);
                    }
                }
            }
            
            // Exchange neighbor information
            std::vector<int> sendCounts(size);
            for (int i = 0; i < size; i++) {
                sendCounts[i] = neighborsToSend[i].size();
            }
            
            std::vector<int> recvCounts(size);
            MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
            
            std::vector<int> sendDisplacements(size);
            std::vector<int> recvDisplacements(size);
            sendDisplacements[0] = 0;
            recvDisplacements[0] = 0;
            
            for (int i = 1; i < size; i++) {
                sendDisplacements[i] = sendDisplacements[i-1] + sendCounts[i-1];
                recvDisplacements[i] = recvDisplacements[i-1] + recvCounts[i-1];
            }
            
            int totalSend = sendDisplacements.back() + sendCounts.back();
            int totalRecv = recvDisplacements.back() + recvCounts.back();
            
            std::vector<int> sendBuffer(totalSend);
            for (int i = 0; i < size; i++) {
                std::copy(neighborsToSend[i].begin(), neighborsToSend[i].end(), 
                          sendBuffer.begin() + sendDisplacements[i]);
            }
            
            std::vector<int> recvBuffer(totalRecv);
            MPI_Alltoallv(sendBuffer.data(), sendCounts.data(), sendDisplacements.data(), MPI_INT,
                         recvBuffer.data(), recvCounts.data(), recvDisplacements.data(), MPI_INT,
                         MPI_COMM_WORLD);
            
            // Process received neighbors
            for (int i = 0; i < totalRecv; i++) {
                int vertex = recvBuffer[i];
                // Check if this vertex is local and not already visited
                for (int localVertex : localVertices) {
                    if (localVertex == vertex && level[vertex] == INFINITY) {
                        level[vertex] = currentLevel + 1;
                        break;
                    }
                }
            }
            
            // Synchronize levels across all processors
            MPI_Allreduce(MPI_IN_PLACE, level.data(), numVertices, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            
            // Move to next level
            currentLevel++;
        }
    }
    
    // Print BFS results
    void printResults() {
        if (rank == 0) {
            std::cout << "\nBFS Results from source vertex " << sourceVertex << ":" << std::endl;
            for (int i = 0; i < numVertices; i++) {
                if (level[i] == INFINITY) {
                    std::cout << "Vertex " << i << ": Unreachable" << std::endl;
                } else {
                    std::cout << "Vertex " << i << ": Level " << level[i] << std::endl;
                }
            }
        }
    }
};

int main(int argc, char** argv) {
    int rank, size;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Default parameters
    int numVertices = 5;  // Default number of vertices
    int sourceVertex = 0;  // Default source vertex
    int graphType = 1;     // 0: File, 1: Manual, 2: Interactive
    std::string filename = "";
    
    // Parse command line arguments if provided
    if (argc > 1) {
        numVertices = std::atoi(argv[1]);
    }
    
    if (argc > 2) {
        sourceVertex = std::atoi(argv[2]);
    }
    
    if (argc > 3) {
        graphType = std::atoi(argv[3]);
    }
    
    if (argc > 4 && graphType == 0) {
        filename = argv[4];
    }
    
    // Validate input
    if (numVertices <= 0 || sourceVertex < 0 || sourceVertex >= numVertices) {
        if (rank == 0) {
            std::cout << "Invalid parameters. Using defaults: numVertices=" << numVertices 
                      << ", sourceVertex=" << sourceVertex << std::endl;
        }
        numVertices = std::max(1, numVertices);
        sourceVertex = std::min(std::max(0, sourceVertex), numVertices - 1);
    }
    
    // Create graph
    Graph graph(numVertices, sourceVertex, rank, size);
    
    // Define graph according to type
    if (graphType == 0) {  // From file
        if (!filename.empty()) {
            graph.loadFromFile(filename);
        } else {
            // Use a default example if no file is provided
            if (rank == 0) {
                std::cout << "No filename provided, using default graph" << std::endl;
            }
            
            // Define a default graph (can be modified as needed)
            std::vector<std::pair<int, int>> defaultEdges = {
                {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5},
                {0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}
            };
            graph.defineGraph(defaultEdges);
        }
    } else if (graphType == 1) {  // Manual definition
        // Define your own graph here by modifying these edges
        std::vector<std::pair<int, int>> customEdges = {
            {0,1},{0,4},{1,4},{1,2},{4,3},{2,3},{2,4}
        };
        graph.defineGraph(customEdges);
    } else {  // Interactive
        graph.inputEdgesInteractively();
    }
    
    // Print the graph structure for verification
    graph.printAdjacencyMatrix();
    
    // Wait for all processes to be ready
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Start timer
    double startTime = MPI_Wtime();
    
    // Perform distributed BFS
    graph.distributedBFS();
    
    // End timer
    double endTime = MPI_Wtime();
    
    if (rank == 0) {
        std::cout << "BFS completed in " << (endTime - startTime) << " seconds" << std::endl;
    }
    
    // Print results
    graph.printResults();
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}