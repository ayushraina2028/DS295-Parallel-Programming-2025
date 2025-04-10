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
    vector<vector< Edge >> AdjacencyList;

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
    const vector<vector< Edge >>& GetAdjancencyList() const {
        return AdjacencyList;
    }

    // Get number of vertices
    int getVertices() const {
        return numberOfVertices;
    }

};

class DeltaStepping {
private:

    const Graph& graph;
    int Delta;
    vector<int> Distances;
    vector<list<int>> Buckets;
    vector<bool> InBucket;

    // Light and Heavy Lists
    vector<vector<Edge>> LightEdges;
    vector<vector<Edge>> HeavyEdges;

    // Function to determine if an edge is light or heavy
    bool isLightEdge(int weight) const {
        return weight <= Delta;
    }

    // Function to find the bucket for the given distance
    int getBucketIndex(int distance) const {
        if(distance == numeric_limits<int>::max()) {
            return -1;
        }
        else return distance/Delta;
    }

    // Function to insert a vertex into correct bucket
    void insertIntoBucket(int Vertex, int NewDistance) {
        
        int BucketIndex = getBucketIndex(NewDistance);
        if(BucketIndex >= Buckets.size()) {
            Buckets.resize(BucketIndex + 1);
        }

        Buckets[BucketIndex].push_back(Vertex);
        InBucket[Vertex] = true;
    }

    // Function to relax edges from a vertex
    void relax(int Vertex, vector<vector<Edge>>& Edges, vector<int>& Changed) {

        for(const Edge& edge : Edges[Vertex]) {
            int Destination = edge.Destination;
            int Weight = edge.weight;

            if(Distances[Vertex] + Weight < Distances[Destination]) {

                // Remove from old bucket if present
                if(InBucket[Destination]) {
                    int OldBucketIndex = getBucketIndex(Distances[Destination]);
                    Buckets[OldBucketIndex].remove(Destination);
                    InBucket[Destination] = false;
                }

                // Update Distances
                Distances[Destination] = Distances[Vertex] + Weight;
                Changed.push_back(Destination);

                insertIntoBucket(Destination, Distances[Destination]);
            }
        }

    }

public:
    DeltaStepping(const Graph& G, int D) : graph(G), Delta(D) {
        int V = graph.getVertices();
        Distances.resize(V, numeric_limits<int>::max());
        InBucket.resize(V, false);

        // Initializing Light and Heavy Edge Arrays
        LightEdges.resize(V);
        HeavyEdges.resize(V);

        const vector<vector<Edge>>& AdjList = graph.GetAdjancencyList();
        for(int Vertex = 0; Vertex < V; Vertex++) {

            for(const Edge& edge : AdjList[Vertex]) {
                if(isLightEdge(edge.weight)) {
                    LightEdges[Vertex].push_back(edge);
                }   
                else {
                    HeavyEdges[Vertex].push_back(edge);
                }
            }

        }
    }

    // Run Delta Stepping From Source Vertex
    vector<int> SingleSourceShortestPath(int Source) {
        
        // Things to be done for Source Vertex
        Distances[Source] = 0;
        Buckets.resize(1);
        Buckets[0].push_back(Source);
        InBucket[Source] = true;

        // Processing Buckets
        for(int i = 0;i < Buckets.size(); i++) {

            while(!Buckets[i].empty()) {
                vector<int> S; // Vertices to be processed

                for(int Vertex : Buckets[i]) {
                    S.push_back(Vertex);
                    InBucket[Vertex] = false;
                }

                Buckets[i].clear();

                // Prcocessing Light Edges First
                vector<int> ChangedVertices;
                for(int vertex : S) {
                    relax(vertex, LightEdges, ChangedVertices);
                }

                // Then Processing Heavy Edges
                for(int vertex : S) {
                    relax(vertex, HeavyEdges, ChangedVertices);
                }
            }

        }

        return Distances;
    }
};

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
    DeltaStepping ds(g, Delta);
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
    return 0;
}
