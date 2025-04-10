#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

int numVertices;
const int Delta = 1;

struct Edge {
    int Source;
    int Destination;
    int Weight;
};

struct DistributedVertexMap {
    unordered_map<int, int> LocalToGlobalMap;
    unordered_map<int, int> GlobalToLocalMap;
};

struct VertexInfo {
    vector<pair<int, int>> VertexEdges;
};

struct VertexDistance {
    int distance;
};

struct DistanceUpdate {
    int globalVertex;
    int newDistance;
};

vector<vector<int>> buckets;

enum EdgeType {
    Light,
    Heavy
};

vector<Edge> parseGraphFile(const string& filename) {
    vector<Edge> edges;
    ifstream file(filename);
    string line;
    int maxVertexId = -1;
    
    while (getline(file, line)) {
        stringstream ss(line);
        int src, dest, weight;
        if (ss >> src >> dest >> weight) {
            edges.push_back({src, dest, weight});
            maxVertexId = max(maxVertexId, max(src, dest));
        }
    }
    
    // Set the global count of vertices
    numVertices = maxVertexId + 1;
    cout << "Number of Vertices: " << numVertices << endl;
    
    return edges;
}

DistributedVertexMap createVertexMapping(const vector<Edge>& GlobalGraph, int rank, int numProcesses) {
    DistributedVertexMap VertexMap;

    if (numProcesses == 1) {
        // For single processor, maintain original vertex IDs
        for (int i = 0; i < numVertices; i++) {
            VertexMap.LocalToGlobalMap[i] = i;
            VertexMap.GlobalToLocalMap[i] = i;
        }
        return VertexMap;
    }

    int LocalIndex = 0;
    for(const Edge& edge : GlobalGraph) {

        /* Processing Source Vertices */
        if(edge.Source % numProcesses == rank and VertexMap.GlobalToLocalMap.count(edge.Source) == 0) {
            VertexMap.GlobalToLocalMap[edge.Source] = LocalIndex;
            VertexMap.LocalToGlobalMap[LocalIndex] = edge.Source;
            LocalIndex++;
        }

        /* Processing Destination Vertices */
        if(edge.Destination % numProcesses == rank and VertexMap.GlobalToLocalMap.count(edge.Destination) == 0) {
            VertexMap.GlobalToLocalMap[edge.Destination] = LocalIndex;
            VertexMap.LocalToGlobalMap[LocalIndex] = edge.Destination;
            LocalIndex++;
        }
    }

    return VertexMap;
}

vector<VertexDistance> initializeDistances(int SourceVertex, DistributedVertexMap& VertexMap, int rank) {
    
    vector<VertexDistance> localDistances(VertexMap.LocalToGlobalMap.size(), {numeric_limits<int>::max()});
    
    /* Check if Source Vertex is on this process */
    if(VertexMap.GlobalToLocalMap.count(SourceVertex) > 0) {
        int localSourceIndex = VertexMap.GlobalToLocalMap[SourceVertex];
        localDistances[localSourceIndex].distance = 0;
    }
    
    return localDistances;

}

void addVertexToBucket(vector<vector<int>>& Buckets, int VertexToAdd, int distance, int Delta) {
    int BucketIndex = static_cast<int>(distance/Delta);

    /* Expand if buckets are not available */
    if(BucketIndex >= Buckets.size()) {
        Buckets.resize(BucketIndex+1);
    }
    Buckets[BucketIndex].push_back(VertexToAdd);
}

int findNextNonEmptyBucket(const vector<vector<int>>& Buckets, int CurrentBucket) {

    for(int i = CurrentBucket+1; i < Buckets.size(); i++) {
        if(Buckets[i].size() > 0) {
            return i;
        }
    }
    return -1;

}

vector<VertexInfo> distributeGraph(const vector<Edge>& GlobalGraph, int rank, int numProcesses, DistributedVertexMap& VertexMap) {

    // cout << "Distributing Graph to Processor " << rank << "\n";
    vector<VertexInfo> LocalGraph;
    if (numProcesses == 1) {
        // For single processor, add all edges 
        for (const Edge& edge : GlobalGraph) {
            int src = edge.Source;
            int dest = edge.Destination;
            int weight = edge.Weight;
            
            // Ensure source vertex exists in local graph
            while (LocalGraph.size() <= src) {
                LocalGraph.push_back(VertexInfo());
            }

            // Ensure destination vertex exists in local graph
            while (LocalGraph.size() <= dest) {
                LocalGraph.push_back(VertexInfo());
            }

            LocalGraph[src].VertexEdges.push_back({dest, weight});
            LocalGraph[dest].VertexEdges.push_back({src, weight});
        }
        
        return LocalGraph;
    }

    // Create a vector to store vertices assigned to this process
    for(const Edge& edge : GlobalGraph) {
        
        // Check if source vertex belongs to this process
        if(edge.Source % numProcesses == rank) {

            // Ensure the vertex exists in local graph
            int LocalSourceIndex = VertexMap.GlobalToLocalMap[edge.Source];
            while(LocalGraph.size() <= LocalSourceIndex) {
                LocalGraph.push_back(VertexInfo());
            }
            LocalGraph[LocalSourceIndex].VertexEdges.push_back({edge.Destination, edge.Weight});
        }
        // Handle Bidirectional Edges
        if(edge.Destination % numProcesses == rank) {
            
            // Ensure the vertex exists in local graph
            int LocalDestinationIndex = VertexMap.GlobalToLocalMap[edge.Destination];
            while(LocalGraph.size() <= LocalDestinationIndex) {
                LocalGraph.push_back(VertexInfo());
            }
            LocalGraph[LocalDestinationIndex].VertexEdges.push_back({edge.Source, edge.Weight});
        }
    }
    // cout << "Graph has been distributed to Processor " << rank << ". Local graph size: " << LocalGraph.size() << "\n";
    return LocalGraph;
}

class CommunicationBuffer {
private:
    vector<DistanceUpdate> updates;
    vector<VertexDistance>& LocalDistances;
    DistributedVertexMap& VertexMap;
    vector<vector<int>>& Buckets;
    int Delta;
    unordered_map<int, int> vertexToBucketMap;  // Maps local vertex ID to bucket index

public:
    CommunicationBuffer(vector<VertexDistance>& localDistances, DistributedVertexMap& vertexMap, vector<vector<int>>& buckets, int delta) 
        : LocalDistances(localDistances), VertexMap(vertexMap), Buckets(buckets), Delta(delta) {}

    void addUpdate(int globalVertex, int newDistance) {
        updates.push_back({globalVertex, newDistance});
        
        // Process local updates immediately
        // Check if this global vertex is local to this process
        if (VertexMap.GlobalToLocalMap.count(globalVertex) > 0) {
            int localIndex = VertexMap.GlobalToLocalMap[globalVertex];
            
            // Always update if the new distance is less than current distance
            if (newDistance < LocalDistances[localIndex].distance) {
                // Print debug information
                // cout << "Processing local update for global vertex " << globalVertex 
                //         << " (local index " << localIndex << ")" 
                //         << " from " << LocalDistances[localIndex].distance 
                //         << " to " << newDistance << endl;
                        
                LocalDistances[localIndex].distance = newDistance;
                // Update bucket assignment
                updateVertexBucket(localIndex, newDistance);
            }
        }
    }

    void clearBuffer() {
        updates.clear();
    }

    void SendUpdates(MPI_Comm comm, int DestinationRank) {
        if(updates.size() > 0) {
            MPI_Send(updates.data(), updates.size(), MPI_2INT, DestinationRank, 0, comm);
        }
    }

    void ReceiveUpdates(MPI_Comm comm, int sourceRank) {
        MPI_Status status;
        int updateCount;
        // Probe for message size
        MPI_Probe(sourceRank, 0, comm, &status);
        MPI_Get_count(&status, MPI_2INT, &updateCount);
        
        // Receive updates
        vector<DistanceUpdate> receivedUpdates(updateCount);
        MPI_Recv(receivedUpdates.data(), updateCount, MPI_2INT, sourceRank, 0, comm, MPI_STATUS_IGNORE);
        
        // Print received updates
        cout << "Received " << updateCount << " updates" << endl;
        
        // Process received updates
        for (const auto& update : receivedUpdates) {
            cout << "Processing update for global vertex " << update.globalVertex 
                    << " with distance " << update.newDistance << endl;
            updateLocalDistance(update);
        }
    }
    
    // Process all updates for all processes (including self)
    // Process all updates for all processes
    void ProcessAllUpdates(MPI_Comm comm) {
        int rank, numProcesses;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &numProcesses);
        
        // Handle single processor case
        if (numProcesses == 1) {
            // Process updates locally
            for (const auto& update : updates) {
                updateLocalDistance(update);
            }
            updates.clear();
            return;
        }
        
        // For multiple processors:
        // First, exchange the number of updates each process has
        int numUpdates = updates.size();
        vector<int> allNumUpdates(numProcesses);
        MPI_Allgather(&numUpdates, 1, MPI_INT, allNumUpdates.data(), 1, MPI_INT, comm);
        
        // Debug output
        // cout << "Rank " << rank << " has " << numUpdates << " updates to send" << endl;
        // for (int i = 0; i < numProcesses; i++) {
        //     cout << "Rank " << rank << " expects " << allNumUpdates[i] << " updates from rank " << i << endl;
        // }
        
        // Process own updates first
        for (const auto& update : updates) {
            if (VertexMap.GlobalToLocalMap.count(update.globalVertex) > 0) {
                updateLocalDistance(update);
            }
        }
        
        // Send updates to other processes
        vector<MPI_Request> sendRequests;
        for (int i = 0; i < numProcesses; i++) {
            if (i != rank && numUpdates > 0) {
                MPI_Request request;
                MPI_Isend(updates.data(), numUpdates, MPI_2INT, i, 0, comm, &request);
                sendRequests.push_back(request);
                // cout << "Rank " << rank << " sent " << numUpdates << " updates to rank " << i << endl;
            }
        }
        
        // Receive updates from other processes
        for (int i = 0; i < numProcesses; i++) {
            if (i != rank && allNumUpdates[i] > 0) {
                vector<DistanceUpdate> receivedUpdates(allNumUpdates[i]);
                MPI_Recv(receivedUpdates.data(), allNumUpdates[i], MPI_2INT, 
                        i, 0, comm, MPI_STATUS_IGNORE);
                
                // cout << "Rank " << rank << " received " << allNumUpdates[i] << " updates from rank " << i << endl;
                
                // Process received updates
                for (const auto& update : receivedUpdates) {
                    updateLocalDistance(update);
                }
            }
        }
        
        // Wait for all sends to complete
        if (!sendRequests.empty()) {
            MPI_Waitall(sendRequests.size(), sendRequests.data(), MPI_STATUSES_IGNORE);
        }
        
        // Clear own updates
        updates.clear();
    }

private:
    void updateLocalDistance(const DistanceUpdate& update) {
        // Check if this global vertex is local to this process
        if (VertexMap.GlobalToLocalMap.count(update.globalVertex) > 0) {
            int localIndex = VertexMap.GlobalToLocalMap[update.globalVertex];
            
            // Always update if the new distance is less than current distance
            if (update.newDistance < LocalDistances[localIndex].distance) {
                // Print debug information
                // cout << "Updating vertex " << update.globalVertex 
                //         << " (local index " << localIndex << ")" 
                //         << " from " << LocalDistances[localIndex].distance 
                //         << " to " << update.newDistance << endl;
                        
                LocalDistances[localIndex].distance = update.newDistance;
                // Update bucket assignment
                updateVertexBucket(localIndex, update.newDistance);
            }
        }
    }
    
    // Modified function to handle bucket transitions
    void updateVertexBucket(int vertexID, int newDistance) {
        int newBucketIndex = static_cast<int>(newDistance/Delta);
        
        // Check if vertex is already in a bucket
        auto it = vertexToBucketMap.find(vertexID);
        if (it != vertexToBucketMap.end()) {
            int oldBucketIndex = it->second;
            
            // If same bucket, no need to change
            if (oldBucketIndex == newBucketIndex) return;
            
            // cout << "Moving vertex " << vertexID << " from bucket " << oldBucketIndex 
            //         << " to bucket " << newBucketIndex << endl;
                    
            // Remove from old bucket
            auto& oldBucket = Buckets[oldBucketIndex];
            oldBucket.erase(std::remove(oldBucket.begin(), oldBucket.end(), vertexID), oldBucket.end());
        } else {
            // cout << "Adding vertex " << vertexID << " to bucket " << newBucketIndex << " for the first time" << endl;
        }
        
        // Ensure we have enough buckets
        if (newBucketIndex >= Buckets.size()) {
            Buckets.resize(newBucketIndex + 1);
        }
        
        // Add to new bucket
        Buckets[newBucketIndex].push_back(vertexID);
        
        // Update bucket mapping
        vertexToBucketMap[vertexID] = newBucketIndex;
    }
};

class DeltaSteppingAlgorithm {
    private:
        vector<VertexInfo>& localGraph;
        vector<VertexDistance>& localDistances;
        vector<vector<int>>& buckets;
        CommunicationBuffer& commBuffer;
        DistributedVertexMap& vertexMap;
        int delta;
        int rank;
        int size;
        MPI_Comm comm;
        
    public:
        DeltaSteppingAlgorithm(
            vector<VertexInfo>& graph,
            vector<VertexDistance>& distances,
            vector<vector<int>>& bucketStructure,
            CommunicationBuffer& buffer,
            DistributedVertexMap& map,
            int deltaValue,
            int processRank,
            MPI_Comm communicator = MPI_COMM_WORLD
        ) : localGraph(graph), 
            localDistances(distances), 
            buckets(bucketStructure),
            commBuffer(buffer),
            vertexMap(map),
            delta(deltaValue),
            rank(processRank),
            comm(communicator) {
                MPI_Comm_size(comm, &size);
                initializeBuckets();
            }

        struct AlgorithmMetrics {
            int totalBucketsProcessed = 0;
            int totalVerticesProcessed = 0;
            int totalLightEdgesProcessed = 0;
            int totalHeavyEdgesProcessed = 0;
            int totalRelaxations = 0;
            int totalCommunicationRounds = 0;
            
            void print(int rank) {
                cout << "Rank " << rank << " metrics:" << endl
                        << "  Buckets processed: " << totalBucketsProcessed << endl
                        << "  Vertices processed: " << totalVerticesProcessed << endl
                        << "  Light edges processed: " << totalLightEdgesProcessed << endl
                        << "  Heavy edges processed: " << totalHeavyEdgesProcessed << endl
                        << "  Total edge relaxations: " << totalRelaxations << endl
                        << "  Communication rounds: " << totalCommunicationRounds << endl;
            }
        } metrics;
    
        void initializeBuckets() {
            for (int localIdx = 0; localIdx < localDistances.size(); localIdx++) {
                if (localDistances[localIdx].distance != numeric_limits<int>::max()) {
                    int distance = localDistances[localIdx].distance;
                    int bucketIdx = distance / delta;
                    
                    if (bucketIdx >= buckets.size()) {
                        buckets.resize(bucketIdx + 1);
                    }
                    buckets[bucketIdx].push_back(localIdx);
                    
                    cout << "Initialized vertex " << localIdx << " (global "
                        << vertexMap.LocalToGlobalMap[localIdx] << ") to bucket " 
                        << bucketIdx << " with distance " << distance << endl;
                }
            }
        }
    
        EdgeType classifyEdge(int weight) {
            return (weight <= delta) ? Light : Heavy;
        }

        int findGlobalNextBucket(int localNextBucket, MPI_Comm comm) {
            int rank, size;
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &size);
            
            // Collect all local next buckets
            vector<int> allBuckets(size);
            MPI_Allgather(&localNextBucket, 1, MPI_INT, allBuckets.data(), 1, MPI_INT, comm);
            
            // Find the minimum valid bucket
            int globalNextBucket = numeric_limits<int>::max();
            for (int bucket : allBuckets) {
                if (bucket != -1 && bucket < globalNextBucket) {
                    globalNextBucket = bucket;
                }
            }
            
            // If no valid buckets found, return -1
            if (globalNextBucket == numeric_limits<int>::max()) {
                return -1;
            }
            
            return globalNextBucket;
        }
    
        int findNextNonEmptyBucket(int startBucket) {
            for (int i = startBucket; i < buckets.size(); i++) {
                if (!buckets[i].empty()) {
                    return i;
                }
            }
            return -1;
        }
    
        void relaxEdge(int sourceLocalIndex, int destGlobalVertex, int edgeWeight) {
            // Get current distance to source
            int currentDistance = localDistances[sourceLocalIndex].distance;
            
            // Skip if source is unreachable
            if (currentDistance == numeric_limits<int>::max())
                return;
                
            // Calculate new tentative distance
            int newDistance = currentDistance + edgeWeight;
            
            // For local vertices, we can update directly
            if (vertexMap.GlobalToLocalMap.count(destGlobalVertex) > 0) {
                int destLocalIndex = vertexMap.GlobalToLocalMap[destGlobalVertex];
                
                // Only update if new distance is smaller
                if (newDistance < localDistances[destLocalIndex].distance) {
                    // Update distance
                    localDistances[destLocalIndex].distance = newDistance;
                    
                    // Calculate bucket for this vertex
                    int bucketIndex = newDistance / delta;
                    
                    // Ensure we have enough buckets
                    if (bucketIndex >= buckets.size()) {
                        buckets.resize(bucketIndex + 1);
                    }
                    
                    // Add vertex to the appropriate bucket
                    buckets[bucketIndex].push_back(destLocalIndex);
                }
            }
            
            // Always add to communication buffer for non-local vertices
            // This ensures updates are properly propagated across processes
            if (vertexMap.GlobalToLocalMap.count(destGlobalVertex) == 0) {
                commBuffer.addUpdate(destGlobalVertex, newDistance);
            }

            metrics.totalRelaxations++;
        }
    
        void processLocalBuckets() {
            int currentBucketIndex = 0;
            int maxIterations = 100000; // Prevent infinite loops
            int iterations = 0;
            
            while (iterations < maxIterations) {
                iterations++;
                
                // Check for empty buckets locally
                int localNextBucket = findNextNonEmptyBucket(currentBucketIndex);
                
                // Share whether each process has more work
                int hasMoreWork = (localNextBucket != -1) ? 1 : 0;
                int totalWork = 0;
                MPI_Allreduce(&hasMoreWork, &totalWork, 1, MPI_INT, MPI_SUM, comm);
                
                // If no process has more work, we're done
                if (totalWork == 0) {
                    cout << "Rank " << rank << " terminating: no more work" << endl;
                    break;
                }
                
                // Find minimum non-empty bucket across all processes
                int globalNextBucket = -1;
                if (localNextBucket != -1) {
                    MPI_Allreduce(&localNextBucket, &globalNextBucket, 1, MPI_INT, MPI_MIN, comm);
                } else {
                    int highValue = numeric_limits<int>::max();
                    MPI_Allreduce(&highValue, &globalNextBucket, 1, MPI_INT, MPI_MIN, comm);
                }
                
                // cout << "Rank " << rank << " iteration " << iterations 
                //      << " processing bucket " << globalNextBucket << endl;
                
                // Check if we have this bucket
                vector<int> vertices;
                if (globalNextBucket < buckets.size() && !buckets[globalNextBucket].empty()) {
                    vertices = std::move(buckets[globalNextBucket]);
                    buckets[globalNextBucket].clear();

                    metrics.totalBucketsProcessed++;
                    metrics.totalVerticesProcessed += vertices.size();
                    
                    // Process light edges first
                    for (int vertex : vertices) {
                        processLightEdges(vertex);
                    }
                    
                    // Process heavy edges next
                    for (int vertex : vertices) {
                        processHeavyEdges(vertex);
                    }
                }

                metrics.totalCommunicationRounds++;
                
                // Ensure all processes have processed their part
                MPI_Barrier(comm);
                
                // Share updates
                commBuffer.ProcessAllUpdates(comm);
                
                // Move to next bucket
                currentBucketIndex = globalNextBucket + 1;
            }
            
            if (iterations >= maxIterations) {
                cout << "Warning: Rank " << rank << " reached iteration limit" << endl;
            }

            metrics.print(rank);
        }
    
    private:
        void processLightEdges(int vertexIndex) {
            // Skip invalid vertices
            if (vertexIndex < 0 || vertexIndex >= localGraph.size())
                return;
                
            int vertexDistance = localDistances[vertexIndex].distance;
            
            // Skip if the vertex is unreachable
            if (vertexDistance == numeric_limits<int>::max())
                return;
                
            // Process all light edges from this vertex
            for (const auto& [destGlobalVertex, weight] : localGraph[vertexIndex].VertexEdges) {
                // Only process light edges
                if (weight <= delta) {
                    // Relax the edge
                    relaxEdge(vertexIndex, destGlobalVertex, weight);
                }
            }

            metrics.totalLightEdgesProcessed += localGraph[vertexIndex].VertexEdges.size();
        }
        
        void processHeavyEdges(int vertexIndex) {
            // Skip invalid vertices
            if (vertexIndex < 0 || vertexIndex >= localGraph.size())
                return;
                
            int vertexDistance = localDistances[vertexIndex].distance;
            
            // Skip if the vertex is unreachable
            if (vertexDistance == numeric_limits<int>::max())
                return;
                
            // Process all heavy edges from this vertex
            for (const auto& [destGlobalVertex, weight] : localGraph[vertexIndex].VertexEdges) {
                // Only process heavy edges
                if (weight > delta) {
                    // Relax the edge
                    relaxEdge(vertexIndex, destGlobalVertex, weight);
                }
            }

            metrics.totalHeavyEdgesProcessed += localGraph[vertexIndex].VertexEdges.size();
        }
    };

void verifyGraphDistribution(MPI_Comm comm) {
    int rank, numProcesses;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &numProcesses);

    // Read Global Graph
    vector<Edge> GlobalGraph = parseGraphFile("SyntheticGraph.txt");

    // Create Vertex Mapping
    DistributedVertexMap VertexMap = createVertexMapping(GlobalGraph, rank, numProcesses);

    // Distribute the graph
    vector<VertexInfo> LocalGraph = distributeGraph(GlobalGraph, rank, numProcesses, VertexMap);

    // Local Graph Information
    int localVertexCount = LocalGraph.size();
    cout << "Processor " << rank << " has " << localVertexCount << " vertices\n";
    int TotalVertices = 0;

    MPI_Reduce(&localVertexCount, &TotalVertices, 1, MPI_INT, MPI_SUM, 0, comm);

    // Rank 0 prints summary
    if(rank == 0) {
        cout << "Total Vertices: " << TotalVertices << "\n";
        cout << "Number of Processes: " << numProcesses << "\n";

        if(numVertices == TotalVertices) {
            cout << "Graph Distribution is Correct\n";
        } else {
            cout << "Graph Distribution is Incorrect\n";
        }
    }

}

void saveParallelDeltaSteppingResults(const vector<int>& Distances, int SourceVertex, const string& filename) {
    ofstream outputFile(filename);
    outputFile << "Shortest distances from source vertex " << SourceVertex << "(Parallel Delta-Stepping Algorithm):\n";

    for(int Vertex = 0; Vertex < Distances.size(); Vertex++) {
        outputFile << "To " << Vertex << ": ";

        if(Distances[Vertex] == numeric_limits<int>::max()) {
            outputFile << "INF";
        } else {
            outputFile << Distances[Vertex];
        }

        if(Vertex < Distances.size()-1) {
            outputFile << "\n";
        }
    }

    outputFile.close();
}

bool compareOutputFiles(const string& dijkstraOutput, const string& deltaSteppingOutput) {
    ifstream dijkstraOutputFile(dijkstraOutput);
    ifstream deltaSteppingOutputFile(deltaSteppingOutput);

    string dijkstraLine, deltaSteppingLine;

    // Skip the first line of both files
    getline(dijkstraOutputFile, dijkstraLine);
    getline(deltaSteppingOutputFile, deltaSteppingLine);    

    // Compare the rest of the lines
    bool same = true;
    int lineNumber = 1;
    while(getline(dijkstraOutputFile, dijkstraLine) && getline(deltaSteppingOutputFile, deltaSteppingLine)) {

        if(dijkstraLine != deltaSteppingLine) {
            same = false;
            cout << "Different output at line " << lineNumber << ":\n";
            cout << "Dijkstra: " << dijkstraLine << "\n";
            cout << "Delta-Stepping: " << deltaSteppingLine << "\n";
            break;
        }


        lineNumber++;
    }

    if(!same) {
        return false;
    }

    if(dijkstraOutputFile.eof() != deltaSteppingOutputFile.eof()) {
        cout << "Both files have different number of lines\n";
        cout << "Dijkstra File EOF: " << dijkstraOutputFile.eof() << "\n";
        cout << "Delta-Stepping File EOF: " << deltaSteppingOutputFile.eof() << "\n";
        return false;
    } 

    return true;
}

void testVertexMapping(MPI_Comm comm) {

    cout << "Testing Vertex Mapping\n";

    int rank, numProcesses;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &numProcesses);

    // Read Global Graph
    vector<Edge> GlobalGraph = parseGraphFile("SyntheticGraph.txt");

    // Create Vertex Mapping
    DistributedVertexMap VertexMap = createVertexMapping(GlobalGraph, rank, numProcesses);

    // Distribute the graph
    vector<VertexInfo> LocalGraph = distributeGraph(GlobalGraph, rank, numProcesses, VertexMap);

    // INitialize Distances
    vector<VertexDistance> Distances = initializeDistances(0, VertexMap, rank);

    /* Verification Checks */
    cout << "Rank: " << rank << "\n";
    cout << "Local Graph Size: " << LocalGraph.size() << "\n";
    cout << "Distances Size: " << Distances.size() << "\n";

    cout << "Testing Vertex Mapping Complete\n";
}

void testBucketStructure() {
    vector<vector<int>> Buckets;

    addVertexToBucket(Buckets, 5, 4, Delta);
    addVertexToBucket(Buckets, 10, 2, Delta);
    addVertexToBucket(Buckets, 15, 7, Delta);

    // Print Bucket Contents
    for(int i = 0;i < Buckets.size(); i++) {
        if(Buckets[i].size() > 0) {
            cout << "Bucket " << i << ": ";
            for(int vertex : Buckets[i]) {
                cout << vertex << " ";
            }
            cout << "\n";
        }
        else {
            cout << "Bucket " << i << " is empty\n";
        }
    }

    // test findNextNonEmptyBucket
    int nextBucket = findNextNonEmptyBucket(Buckets, 1);
    cout << "Next Non-Empty Bucket after 2: " << nextBucket << "\n";
}

void testCommunicationBufferFunctionality(MPI_Comm comm) {
    int rank, numProcesses;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &numProcesses);
    
    // Constants
    int delta = 20;
    int sourceVertex = 0;
    
    // Read global graph
    vector<Edge> globalGraph = parseGraphFile("SyntheticGraph.txt");
    
    // Create vertex mapping
    DistributedVertexMap vertexMap = createVertexMapping(globalGraph, rank, numProcesses);
    
    // Distribute graph
    vector<VertexInfo> localGraph = distributeGraph(globalGraph, rank, numProcesses, vertexMap);
    
    // Initialize distances
    vector<VertexDistance> localDistances = initializeDistances(sourceVertex, vertexMap, rank);
    
    // Initialize buckets
    vector<vector<int>> buckets;
    
    // Print initial distances
    // cout << "Rank " << rank << " - Initial Distances:" << endl;
    // for (size_t i = 0; i < localDistances.size(); ++i) {
    //     cout << "Vertex " << i << " (Global " << vertexMap.LocalToGlobalMap[i] << "): " 
    //          << localDistances[i].distance << endl;
    // }
    
    // Create communication buffer
    CommunicationBuffer commBuffer(
        localDistances,
        vertexMap,
        buckets,
        delta
    );
    
    // Synchronization barrier to ensure all processes are ready
    MPI_Barrier(comm);
    
    // Simulate updates from rank 0
    if (rank == 0) {
        // Simulate finding shorter paths
        vector<pair<int, int>> updates = {
            {1, 2},  // Global vertex 1, new distance 2
            {2, 1},  // Global vertex 2, new distance 1
            {3, 3}   // Global vertex 3, new distance 3
        };
        
        cout << "Rank 0 Sending Updates:" << endl;
        for (const auto& [vertex, distance] : updates) {
            commBuffer.addUpdate(vertex, distance);
            cout << "Added update for vertex " << vertex
                 << " with distance " << distance << endl;
        }
        
        // Send updates to all other processes
        for (int i = 1; i < numProcesses; ++i) {
            cout << "Sending updates to Rank " << i << endl;
            commBuffer.SendUpdates(comm, i);
        }
    }
    
    // Other processes receive and process updates
    if (rank != 0) {
        cout << "Rank " << rank << " waiting to receive updates" << endl;
        // Receive and process updates
        commBuffer.ReceiveUpdates(comm, 0);
    }
    
    // Print updated distances for all processes
    cout << "Rank " << rank << " after processing updates:" << endl;
    // cout << "Updated Distances:" << endl;
    // for (size_t i = 0; i < localDistances.size(); ++i) {
    //     cout << "Vertex " << i << " (Global " << vertexMap.LocalToGlobalMap[i] << "): " 
    //          << localDistances[i].distance << " in process " << rank << endl;
    // }

    
    // Print the bucket contents
    cout << "Rank " << rank << " bucket contents:" << endl;
    // for (size_t i = 0; i < buckets.size(); ++i) {
    //     if (!buckets[i].empty()) {
    //         cout << "Bucket " << i << ": ";
    //         for (int vertexID : buckets[i]) {
    //             cout << vertexID << " (Global " << vertexMap.LocalToGlobalMap[vertexID] << ") ";
    //         }
    //         cout << endl;
    //     }
    // }
    
    // Final synchronization
    MPI_Barrier(comm);
}

// Add this function to gather results from all processes
void gatherAndSaveFinalResults(const vector<VertexDistance>& localDistances, const DistributedVertexMap& vertexMap, int sourceVertex, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // Create array of (globalVertex, distance) pairs from local results
    vector<pair<int, int>> localResults;
    for (int i = 0; i < localDistances.size(); i++) {
        int globalId = vertexMap.LocalToGlobalMap.at(i);
        int distance = localDistances[i].distance;
        localResults.push_back({globalId, distance});
    }
    
    if (rank == 0) {
        // Rank 0 will collect results from all processes
        vector<pair<int, int>> allResults = localResults;
        
        // Receive results from other processes
        for (int i = 1; i < size; i++) {
            MPI_Status status;
            int count;
            
            // First get the count of results
            MPI_Probe(i, 0, comm, &status);
            MPI_Get_count(&status, MPI_2INT, &count);
            
            // Then receive the results
            vector<pair<int, int>> procResults(count);
            MPI_Recv(procResults.data(), count, MPI_2INT, i, 0, comm, MPI_STATUS_IGNORE);
            
            // Add to combined results
            allResults.insert(allResults.end(), procResults.begin(), procResults.end());
        }
        
        // Process all results to get minimum distances
        vector<int> finalDistances(numVertices, numeric_limits<int>::max());
        for (const auto& [vertex, dist] : allResults) {
            finalDistances[vertex] = min(finalDistances[vertex], dist);
        }
        
        // Save to file
        saveParallelDeltaSteppingResults(finalDistances, sourceVertex, "ParallelDeltaSteppingOutput.txt");
    }
    else {
        // Other processes send their results to rank 0
        MPI_Send(localResults.data(), localResults.size(), MPI_2INT, 0, 0, comm);
    }
}

void testLocalBucketProcessing(MPI_Comm comm) {
    int rank, numProcesses;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &numProcesses);

    // Constants
    int delta = 1;
    int sourceVertex = 0;

    // Read global graph
    vector<Edge> globalGraph = parseGraphFile("SyntheticGraph.txt");

    // Create vertex mapping
    DistributedVertexMap vertexMap = createVertexMapping(globalGraph, rank, numProcesses);

    // Distribute graph
    vector<VertexInfo> localGraph = distributeGraph(globalGraph, rank, numProcesses, vertexMap);

    // Initialize distances
    vector<VertexDistance> localDistances = initializeDistances(sourceVertex, vertexMap, rank);

    // Initialize buckets
    vector<vector<int>> buckets;

    // Create communication buffer
    CommunicationBuffer commBuffer(
        localDistances,
        vertexMap,
        buckets,
        delta
    );

    // Create Delta Stepping Algorithm instance
    DeltaSteppingAlgorithm deltaStepping(
        localGraph,
        localDistances,
        buckets,
        commBuffer,
        vertexMap,
        delta,
        rank,
        comm
    );

    // Print initial state
    cout << "Rank " << rank << " Initial State:" << endl;
    cout << "Local Graph Size: " << localGraph.size() << endl;
    

    double StartTime = MPI_Wtime();

    // Process local buckets
    deltaStepping.processLocalBuckets();

    double EndTime = MPI_Wtime();
    double elapsedTime = EndTime - StartTime;
    cout << "Time taken by Processor " << rank << ": " << elapsedTime << " seconds\n";

    char filename[100];
    sprintf(filename,"results_%d.txt",numProcesses);
    
    vector<double> allTimes(numProcesses);
    MPI_Gather(&elapsedTime, 1, MPI_DOUBLE, allTimes.data(), 1, MPI_DOUBLE, 0, comm);

    if(rank == 0) {
        ofstream timeFile(filename,ios::app);
        timeFile << "Execution with " << numProcesses << " processors\n";

        double maxTime = 0;
        double totalTime = 0;
        for(int i = 0;i < numProcesses; i++) {
            timeFile << " Process " << i << ": " << allTimes[i] << " seconds\n";
            totalTime += allTimes[i];
            maxTime = max(maxTime, allTimes[i]);
        }

        timeFile << "Maximum Time: " << maxTime << " seconds\n";
        timeFile << "Average Time Per Process: " << totalTime/numProcesses << " seconds\n";
        timeFile << "--------------------------------------------\n";

        timeFile.close();
    }

    // Add this time to end of file named execution_time.txt, Mention time taken by 

    // Gather and save results from all processes
    gatherAndSaveFinalResults(localDistances, vertexMap, sourceVertex, comm);
    
    // Synchronization
    MPI_Barrier(comm);
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    
    // verifyGraphDistribution(MPI_COMM_WORLD);
    // testVertexMapping(MPI_COMM_WORLD);
    // testBucketStructure();
    // testCommunicationBufferFunctionality(MPI_COMM_WORLD);
    testLocalBucketProcessing(MPI_COMM_WORLD);

    bool isResultMatch = compareOutputFiles("dijkstra_output.txt", "ParallelDeltaSteppingOutput.txt");
    if(isResultMatch) {
        cout << "MATCH ✅\n";
    }
    else {
        cout << "NO MATCH ❌\n";
    }

    MPI_Finalize();
    return 0;
}