#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

// Structure to represent edges in the graph
struct Edge {
    int target;
    int weight;
    
    Edge(int t, int w) : target(t), weight(w) {}
};

// Structure to represent a graph
class Graph {
private:
    int V; // Number of vertices
    vector<vector<Edge>> adj; // Adjacency list
    
public:
    // Constructor
    Graph(int vertices) : V(vertices) {
        adj.resize(vertices);
    }
    
    // Function to add an edge to the graph
    void addEdge(int src, int dest, int weight) {
        adj[src].push_back(Edge(dest, weight));
    }
    
    // Get the adjacency list
    const vector<vector<Edge>>& getAdjList() const {
        return adj;
    }
    
    // Get number of vertices
    int getV() const {
        return V;
    }
};

// Implementation of Dijkstra's Algorithm
class Dijkstra {
private:
    const Graph& graph;
    
public:
    Dijkstra(const Graph& g) : graph(g) {}
    
    // Comparator for priority queue
    struct CompareDistance {
        bool operator()(const pair<int, int>& p1, const pair<int, int>& p2) {
            return p1.second > p2.second; // Min heap based on distance
        }
    };
    
    // Run Dijkstra's algorithm from source vertex
    vector<int> shortestPath(int source) {
        int V = graph.getV();
        vector<int> dist(V, numeric_limits<int>::max());
    
        // Min-heap priority queue
        priority_queue<pair<int, int>, vector<pair<int, int>>, CompareDistance> pq;
    
        // Initialize source distance
        dist[source] = 0;
        pq.push({source, 0});
    
        while (!pq.empty()) {
            int u = pq.top().first;
            int d = pq.top().second;
            pq.pop();
    
            // Ignore outdated distances
            if (d > dist[u]) continue;
    
            // Relax edges
            for (const Edge& edge : graph.getAdjList()[u]) {
                int v = edge.target;
                int weight = edge.weight;
    
                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.push({v, dist[v]});
                }
            }
        }
        return dist;
    }
    
};

high_resolution_clock::time_point getTime() {
    return high_resolution_clock::now();
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
        g.addEdge(u, v, w);
    }

    inFile.close();
    cout << "Edges added\n";

    // Run Dijkstra's algorithm from source 0

    high_resolution_clock::time_point t1 = getTime();

    cout << "Running Dijkstra's algorithm...\n";
    Dijkstra dijkstra(g);
    vector<int> shortestDistances = dijkstra.shortestPath(0);
    cout << "Dijkstra's algorithm completed\n";

    high_resolution_clock::time_point t2 = getTime();
    auto duration = duration_cast<milliseconds>(t2 - t1).count();
    cout << "Time taken: " << duration << " milliseconds\n";
    
    // Write results to file

    cout << "Writing results to file...\n";
    ofstream outFile("dijkstra_output.txt");
    if (!outFile) {
        cerr << "Error opening output file!\n";
        return 1;
    }

    outFile << "Shortest distances from source vertex 0 (Dijkstra's Algorithm):\n";
    for (int i = 0; i < shortestDistances.size(); i++) {
        outFile << "To " << i << ": ";
        if (shortestDistances[i] == numeric_limits<int>::max())
            outFile << "INF";
        else
            outFile << shortestDistances[i];
        outFile << endl;
    }

    outFile.close();

    cout << "Results written to file\n";
    return 0;
}

