#include <bits/stdc++.h>
using namespace std;

constexpr int My_Infinity = numeric_limits<int>::infinity();

class Graph {
public:
    int numberOfVertices;
    vector<vector<pair<int, int>>> AdjacencyList;

    Graph(int numberOfVertices) {

        this->numberOfVertices = numberOfVertices;
        AdjacencyList.resize(numberOfVertices);
    
    }

    void addEdge(int Source, int Destination, int Weight) { // Assuming Graph is undirected

        if(Source >= 0 && Source < numberOfVertices && Destination >= 0 && Destination < numberOfVertices) {
            AdjacencyList[Source].push_back({Destination, Weight});
            AdjacencyList[Destination].push_back({Source, Weight});
        }
        else {
            cout << "Warning: Invalid Edge (" << Source << ", " << Destination << ") Ignored." << endl;
        }

    }
};

class DeltaStepping {
public:
    Graph graph;
    int Delta;
    vector<int> Distances;
    vector<set<int>> Buckets;
    unordered_map<int, int> bucketIndex;
    
    DeltaStepping(Graph G, int Delta) : graph(G) {
        this->Delta = Delta;
        Distances.resize(graph.numberOfVertices, My_Infinity);
        Buckets.resize(graph.numberOfVertices);
    }        

    void relaxEdge(int v, int newDistance) {

    }

};

int main() {
    int numberOfVertices = 6;
    vector<vector<int>> Edges = {
        
    };
}