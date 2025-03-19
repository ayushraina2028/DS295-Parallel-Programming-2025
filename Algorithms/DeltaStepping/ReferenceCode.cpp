#include <iostream>
#include <vector>
#include <queue>
#include <list>
#include <limits>
#include <algorithm>

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

// Implementation of Delta-Stepping Algorithm
class DeltaStepping {
private:
    const Graph& graph;
    int delta;  // The delta parameter
    vector<int> dist;  // Distance array
    vector<list<int>> buckets;  // Buckets
    vector<bool> inBucket;  // Track if a node is in a bucket
    
    // Divide adjacency list into light and heavy edges
    vector<vector<Edge>> light;
    vector<vector<Edge>> heavy;
    
    // Function to determine if an edge is light or heavy
    bool isLight(int weight) const {
        return weight <= delta;
    }
    
    // Find the bucket for a given distance
    int getBucketIndex(int distance) const {
        if (distance == numeric_limits<int>::max())
            return -1;
        return distance / delta;
    }
    
    // Insert a vertex into appropriate bucket
    void insertIntoBucket(int v, int newDist) {
        int bucketIdx = getBucketIndex(newDist);
        
        if (bucketIdx >= buckets.size()) {
            buckets.resize(bucketIdx + 1);
        }
        
        buckets[bucketIdx].push_back(v);
        inBucket[v] = true;
    }
    
    // Relax edges from a vertex
    void relax(int v, vector<vector<Edge>>& edges, vector<int>& changed) {
        for (const Edge& e : edges[v]) {
            int u = e.target;
            int weight = e.weight;
            
            if (dist[v] + weight < dist[u]) {
                // Remove from old bucket if present
                if (inBucket[u]) {
                    int oldBucketIdx = getBucketIndex(dist[u]);
                    buckets[oldBucketIdx].remove(u);
                    inBucket[u] = false;
                }
                
                // Update distance
                dist[u] = dist[v] + weight;
                changed.push_back(u);
                
                // Add to new bucket
                insertIntoBucket(u, dist[u]);
            }
        }
    }
    
public:
    DeltaStepping(const Graph& g, int d) : graph(g), delta(d) {
        int V = graph.getV();
        dist.resize(V, numeric_limits<int>::max());
        inBucket.resize(V, false);
        
        // Initialize light and heavy edge arrays
        light.resize(V);
        heavy.resize(V);
        
        // Classify edges as light or heavy
        const auto& adjList = graph.getAdjList();
        for (int v = 0; v < V; v++) {
            for (const Edge& e : adjList[v]) {
                if (isLight(e.weight)) {
                    light[v].push_back(e);
                } else {
                    heavy[v].push_back(e);
                }
            }
        }
    }
    
    // Run Delta-Stepping from source vertex
    vector<int> shortestPath(int source) {
        // Initialize distance to source as 0
        dist[source] = 0;
        
        // Insert source into first bucket
        buckets.resize(1);
        buckets[0].push_back(source);
        inBucket[source] = true;
        
        // Process buckets
        for (int i = 0; i < buckets.size(); i++) {
            // Process all vertices in current bucket until it's empty
            while (!buckets[i].empty()) {
                vector<int> S;  // Vertices to be processed
                
                // Move all vertices from bucket i to S
                for (int v : buckets[i]) {
                    S.push_back(v);
                    inBucket[v] = false;
                }
                buckets[i].clear();
                
                // First process light edges
                vector<int> changedVertices;
                for (int v : S) {
                    relax(v, light, changedVertices);
                }
                
                // Then process heavy edges
                for (int v : S) {
                    relax(v, heavy, changedVertices);
                }
            }
        }
        
        return dist;
    }
};

// Example usage
int main() {
    // Create a graph with 6 vertices
    Graph g(6);
    
    // Add edges
    g.addEdge(0, 1, 2);
    g.addEdge(0, 2, 4);
    g.addEdge(1, 2, 1);
    g.addEdge(1, 3, 7);
    g.addEdge(2, 4, 3);
    g.addEdge(3, 5, 1);
    g.addEdge(4, 3, 2);
    g.addEdge(4, 5, 5);
    
    // Set delta parameter (a good choice is often the average edge weight)
    int delta = 3;
    
    // Run Delta-Stepping algorithm
    DeltaStepping ds(g, delta);
    vector<int> shortestDistances = ds.shortestPath(0);
    
    // Print results
    cout << "Shortest distances from source vertex 0:\n";
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