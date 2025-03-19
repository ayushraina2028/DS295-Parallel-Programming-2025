#include <bits/stdc++.h>
using namespace std;

int main() {
    int numberOfvertices = 6;
    vector<pair<int, int>> Edges = {
        {0,1},{1,2},{1,5},{2,5},{2,3},{3,5},{3,4},{5,4}
    };

    vector<vector<int>> adj(numberOfvertices);
    for(auto edge : Edges) { /* Assuming Graph is Undirected */
        adj[edge.first].push_back(edge.second);
        adj[edge.second].push_back(edge.first);
    }

    
}