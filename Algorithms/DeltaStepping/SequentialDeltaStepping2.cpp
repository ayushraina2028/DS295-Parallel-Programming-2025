#include <bits/stdc++.h>
using namespace std;
using namespace chrono;

// Structure for Edge
struct Edge {
    int Destination;
    int weight;

    Edge(int Destination, int weight) {
        this->Destination = Destination;
        this->weight = weight;
    }
};

// Structure for Graph
class Graph {
private:
    int numberOfVertices;
    vector<vector<Edge>> AdjacencyList;

public:
    Graph(int Vertices) {
        numberOfVertices = Vertices;
        AdjacencyList.resize(Vertices);
    }

    // Function to addEdge
    void AddEdge(int Source, int Destination, int Weight, bool undirected = true) {
        AdjacencyList[Source].push_back(Edge(Destination, Weight));
        if(undirected) {
            AdjacencyList[Destination].push_back(Edge(Source, Weight));
        }
    }

    // Get the adjacency List
    const vector<vector<Edge>>& GetAdjancencyList() const {
        return AdjacencyList;
    }

    // Get number of vertices
    int getVertices() const {
        return numberOfVertices;
    }
};

class ImprovedDeltaStepping {
private:
    const Graph& graph;
    int Delta;
    vector<int> Distances;
    vector<vector<int>> Buckets;  // Using vector instead of list for better cache locality
    vector<int> BucketIndex;      // Track which bucket a vertex is in (-1 if none)

    // Light and Heavy Lists
    vector<vector<Edge>> LightEdges;
    vector<vector<Edge>> HeavyEdges;

    // Utility function to determine if an edge is light or heavy
    bool isLightEdge(int weight) const {
        return weight <= Delta;
    }

    // Function to find the bucket for the given distance
    int getBucketIndex(int distance) const {
        if(distance == numeric_limits<int>::max()) {
            return -1;
        }
        return distance / Delta;
    }

    // Function to insert a vertex into correct bucket
    void insertIntoBucket(int Vertex, int NewDistance) {
        int bucketIdx = getBucketIndex(NewDistance);
        if(bucketIdx >= Buckets.size()) {
            Buckets.resize(bucketIdx + 1);
        }
        
        Buckets[bucketIdx].push_back(Vertex);
        BucketIndex[Vertex] = bucketIdx;
    }

    // Function to relax edges from a vertex
    void relax(int Vertex, const vector<vector<Edge>>& Edges, vector<int>& Changed) {
        for(const Edge& edge : Edges[Vertex]) {
            int Destination = edge.Destination;
            int Weight = edge.weight;
            int newDist = Distances[Vertex] + Weight;

            if(newDist < Distances[Destination]) {
                // If already in a bucket, remove it
                if(BucketIndex[Destination] != -1) {
                    // We don't actually remove from the bucket vector here
                    // We'll filter out these vertices when processing buckets
                    BucketIndex[Destination] = -1;
                }

                // Update distance
                Distances[Destination] = newDist;
                Changed.push_back(Destination);
                
                // Insert into new bucket
                insertIntoBucket(Destination, newDist);
            }
        }
    }

public:
    ImprovedDeltaStepping(const Graph& G, int D) : graph(G), Delta(D) {
        int V = graph.getVertices();
        Distances.resize(V, numeric_limits<int>::max());
        BucketIndex.resize(V, -1);  // -1 means not in any bucket

        // Auto-tune Delta if not set properly
        if (Delta <= 0) {
            // Calculate average edge weight
            long long sum = 0;
            int count = 0;
            const auto& adjList = graph.GetAdjancencyList();
            for(int v = 0; v < V; v++) {
                for(const auto& edge : adjList[v]) {
                    sum += edge.weight;
                    count++;
                }
            }
            Delta = (count > 0) ? max(1, (int)(sum / count)) : 1;
        }

        // Initialize Light and Heavy Edge Arrays
        LightEdges.resize(V);
        HeavyEdges.resize(V);

        const auto& AdjList = graph.GetAdjancencyList();
        for(int Vertex = 0; Vertex < V; Vertex++) {
            for(const Edge& edge : AdjList[Vertex]) {
                if(isLightEdge(edge.weight)) {
                    LightEdges[Vertex].push_back(edge);
                } else {
                    HeavyEdges[Vertex].push_back(edge);
                }
            }
        }
    }

    // Improved Single Source Shortest Path function
    vector<int> SingleSourceShortestPath(int Source) {
        // Reset data structures for Source
        fill(Distances.begin(), Distances.end(), numeric_limits<int>::max());
        fill(BucketIndex.begin(), BucketIndex.end(), -1);
        Buckets.clear();
        
        // Initialize source vertex
        Distances[Source] = 0;
        Buckets.resize(1);
        Buckets[0].push_back(Source);
        BucketIndex[Source] = 0;

        // Process buckets
        for(int i = 0; i < Buckets.size(); i++) {
            // Continue processing current bucket until it's empty
            while(!Buckets[i].empty()) {
                // Get all vertices in current bucket
                vector<int> currentVertices;
                for(int v : Buckets[i]) {
                    // Only consider vertices that are still in this bucket
                    if(BucketIndex[v] == i) {
                        currentVertices.push_back(v);
                        BucketIndex[v] = -1;  // Mark as removed from bucket
                    }
                }
                Buckets[i].clear();  // Clear the bucket now that we have the vertices
                
                // First process light edges
                vector<int> changedVertices;
                for(int v : currentVertices) {
                    relax(v, LightEdges, changedVertices);
                }
                
                // Then process heavy edges
                for(int v : currentVertices) {
                    relax(v, HeavyEdges, changedVertices);
                }
            }
        }
        
        return Distances;
    }
};

// Timing utility function
void measureTime(const Graph& g, int source, int delta) {
    cout << "Testing with Delta = " << delta << endl;
    
    // Measure Delta-Stepping time
    auto start = high_resolution_clock::now();
    ImprovedDeltaStepping deltaAlgo(g, delta);
    vector<int> dsDist = deltaAlgo.SingleSourceShortestPath(source);
    auto end = high_resolution_clock::now();
    auto dsTime = duration_cast<milliseconds>(end - start).count();
    
    // Run Dijkstra for comparison
    start = high_resolution_clock::now();
    
    // Dijkstra implementation
    vector<int> dijkDist(g.getVertices(), numeric_limits<int>::max());
    dijkDist[source] = 0;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    pq.push({0, source});
    
    while(!pq.empty()) {
        int u = pq.top().second;
        int d = pq.top().first;
        pq.pop();
        
        if(d > dijkDist[u]) continue;
        
        for(const Edge& e : g.GetAdjancencyList()[u]) {
            int v = e.Destination;
            int w = e.weight;
            
            if(dijkDist[u] + w < dijkDist[v]) {
                dijkDist[v] = dijkDist[u] + w;
                pq.push({dijkDist[v], v});
            }
        }
    }
    
    end = high_resolution_clock::now();
    auto dijkTime = duration_cast<milliseconds>(end - start).count();
    
    // Verify correctness
    bool correct = true;
    for(int i = 0; i < g.getVertices(); i++) {
        if(dijkDist[i] != dsDist[i]) {
            correct = false;
            cout << "Mismatch at vertex " << i << ": Dijkstra=" << dijkDist[i] 
                 << ", Delta-Stepping=" << dsDist[i] << endl;
            break;
        }
    }
    
    cout << "Delta-Stepping time: " << dsTime << " ms" << endl;
    cout << "Dijkstra time: " << dijkTime << " ms" << endl;
    cout << "Results are " << (correct ? "correct" : "incorrect") << endl;
    cout << "Speedup: " << (dijkTime > 0 ? (double)dijkTime / dsTime : 0) << "x" << endl;
    cout << "-----------------------------------" << endl;
}

int main() {
    ifstream inFile("graph.txt");
    if (!inFile) {
        cerr << "Error opening input file!\n";
        return 1;
    }

    int maxVertex = 0;
    int u, v, w;

    // Determine the number of vertices dynamically
    while (inFile >> u >> v >> w) {
        maxVertex = max(maxVertex, max(u, v));
    }
    
    cout << "Number of vertices: " << maxVertex + 1 << endl;

    // Reset file pointer to read edges again
    inFile.clear();
    inFile.seekg(0, ios::beg);

    // Create graph with correct size
    Graph g(maxVertex + 1);
    cout << "Graph created\n";
    cout << "Adding edges...\n";

    // Read edges and directly add them to the graph
    while (inFile >> u >> v >> w) {
        g.AddEdge(u, v, w);
    }

    inFile.close();
    cout << "Edges added\n";

    // Define Delta value
    int Delta = 50;
    cout << "Delta set to " << Delta << "\n";

    // Run Delta-Stepping algorithm from source 0
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    cout << "Running Delta-Stepping algorithm...\n";
    ImprovedDeltaStepping ds(g, Delta);
    int SourceVertex = 0;
    vector<int> shortestDistances = ds.SingleSourceShortestPath(SourceVertex);
    cout << "Delta-Stepping algorithm completed\n";

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t2 - t1).count();
    cout << "Time taken: " << duration << " milliseconds\n";
    
    // Write results to file
    cout << "Writing results to file...\n";
    ofstream outFile("delta_stepping_output.txt");
    if (!outFile) {
        cerr << "Error opening output file!\n";
        return 1;
    }

    outFile << "Shortest distances from source vertex " << SourceVertex << " (Delta-Stepping Algorithm):\n";
    for (int i = 0; i < shortestDistances.size(); i++) {
        outFile << "To " << i << ": ";
        if (shortestDistances[i] == numeric_limits<int>::max())
            outFile << "INF";
        else
            outFile << shortestDistances[i];
        outFile << '\n';
    }

    outFile.close();

    cout << "Results written to file\n";

    // Measure time for different Delta values
    measureTime(g, SourceVertex, 5);
    measureTime(g, SourceVertex, 10);
    measureTime(g, SourceVertex, 20);
    measureTime(g, SourceVertex, 50);
    return 0;
}