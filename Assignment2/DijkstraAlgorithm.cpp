#include <bits/stdc++.h>
using namespace std;
using namespace chrono;

// Edge representation
struct Edge {
    int target, weight;
    Edge(int t, int w) : target(t), weight(w) {}
};

// Graph class
class Graph {
private:
    int V; 
    vector<vector<Edge>> adj; 

public:
    Graph(int vertices) : V(vertices) {
        adj.resize(vertices);
    }
    
    void addEdge(int src, int dest, int weight) {
        adj[src].push_back(Edge(dest, weight));
    }

    const vector<vector<Edge>>& getAdjList() const {
        return adj;
    }

    int getV() const {
        return V;
    }
};

// Dijkstra's Algorithm
class Dijkstra {
private:
    const Graph& graph;
public:
    Dijkstra(const Graph& g) : graph(g) {}

    vector<int> shortestPath(int source) {
        int V = graph.getV();
        vector<int> dist(V, numeric_limits<int>::max());
        
        // Min-heap: (distance, vertex)
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;

        dist[source] = 0;
        pq.push({0, source});  

        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();

            // Ignore outdated entries
            if (d > dist[u]) continue;

            for (const Edge& edge : graph.getAdjList()[u]) {
                int v = edge.target;
                int weight = edge.weight;

                // Relax edge if a shorter path is found
                if (dist[u] + weight < dist[v]) {
                    dist[v] = dist[u] + weight;
                    pq.push({dist[v], v});  // Push updated (distance, vertex)
                }
            }
        }
        return dist;
    }
};

int main() {
    ifstream inFile("SyntheticGraph.txt");
    if (!inFile) {
        cerr << "Error opening input file!\n";
        return 1;
    }

    int u, v, w, maxVertex = 0;
    vector<tuple<int, int, int>> edges;

    while (inFile >> u >> v >> w) {
        maxVertex = max({maxVertex, u, v});
        edges.emplace_back(u, v, w);
    }
    inFile.close();

    cout << "Number of vertices: " << maxVertex + 1 << endl;

    Graph g(maxVertex + 1);
    for (auto [u, v, w] : edges) {
        g.addEdge(u, v, w);
        g.addEdge(v, u, w);  // Bidirectional edge
    }

    cout << "Edges added\n";

    Dijkstra dijkstra(g);

    auto t1 = high_resolution_clock::now();
    vector<int> shortestDistances = dijkstra.shortestPath(0);
    auto t2 = high_resolution_clock::now();

    cout << "Time taken: " 
         << duration_cast<milliseconds>(t2 - t1).count() 
         << " ms\n";

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
        if(i < shortestDistances.size()-1) outFile << '\n';
    }

    outFile.close();
    cout << "Results written to file\n";
    return 0;
}
