#include <iostream>
#include <vector>
#include <queue>
#include <limits>

using namespace std;

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
        vector<bool> visited(V, false);
        
        // Priority queue to store vertices that need to be processed
        // pair<vertex, distance>
        priority_queue<pair<int, int>, vector<pair<int, int>>, CompareDistance> pq;
        
        // Initialize source distance as 0
        dist[source] = 0;
        pq.push(make_pair(source, 0));
        
        while (!pq.empty()) {
            // Extract vertex with minimum distance
            int u = pq.top().first;
            pq.pop();
            
            // If already processed, skip
            if (visited[u]) {
                continue;
            }
            
            // Mark vertex as processed
            visited[u] = true;
            
            // Process all adjacent vertices
            const auto& adjList = graph.getAdjList();
            for (const Edge& edge : adjList[u]) {
                int v = edge.target;
                int weight = edge.weight;
                
                // If there is a shorter path to v through u
                if (!visited[v] && dist[u] != numeric_limits<int>::max() && 
                    dist[u] + weight < dist[v]) {
                    // Update distance of v
                    dist[v] = dist[u] + weight;
                    pq.push(make_pair(v, dist[v]));
                }
            }
        }
        
        return dist;
    }
};

// Example usage with the same graph as Delta-Stepping
int main() {
    // Create a graph with 6 vertices
    Graph g(6);
    
    // Add edges (same as in Delta-Stepping example)
    g.addEdge(0, 1, 2);
    g.addEdge(0, 2, 4);
    g.addEdge(1, 2, 1);
    g.addEdge(1, 3, 7);
    g.addEdge(2, 4, 3);
    g.addEdge(3, 5, 1);
    g.addEdge(4, 3, 2);
    g.addEdge(4, 5, 5);
    
    // Run Dijkstra's algorithm
    Dijkstra dijkstra(g);
    vector<int> shortestDistances = dijkstra.shortestPath(0);
    
    // Print results
    cout << "Shortest distances from source vertex 0 (Dijkstra's Algorithm):\n";
    for (int i = 0; i < shortestDistances.size(); i++) {
        cout << "To " << i << ": ";
        if (shortestDistances[i] == numeric_limits<int>::max())
            cout << "INF";
        else
            cout << shortestDistances[i];
        cout << endl;
    }
    
    return 0;
}