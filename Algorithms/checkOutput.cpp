#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

using namespace std;

// Function to read distances from file and store in a map
map<int, int> readDistances(const string& filename) {
    ifstream inFile(filename);
    if (!inFile) {
        cerr << "Error opening file: " << filename << "\n";
        exit(1);
    }

    map<int, int> distances;
    string line;
    
    // Skip the first line (header)
    getline(inFile, line);

    // Read distances
    int node, distance;
    while (getline(inFile, line)) {
        istringstream iss(line);
        string temp;
        iss >> temp; // Read "To"
        iss >> node; // Read node number
        iss >> temp; // Read ":"
        if (iss >> distance) {
            distances[node] = distance;
        } else {
            distances[node] = -1; // Store -1 for "INF"
        }
    }

    inFile.close();
    return distances;
}

int main() {
    // Read distances from both files
    map<int, int> dijkstraDistances = readDistances("Dijkstra/dijkstra_output.txt");
    map<int, int> deltaSteppingDistances = readDistances("DeltaStepping/delta_stepping_output.txt");

    // Compare the results
    bool isSame = true;
    for (const auto& X : dijkstraDistances) {
        auto node = X.first;
        int dijkstraDist = X.second;
        if (deltaSteppingDistances.find(node) == deltaSteppingDistances.end()) {
            cout << "Mismatch: Node " << node << " is missing in Delta-Stepping output\n";
            isSame = false;
            continue;
        }

        int deltaDist = deltaSteppingDistances[node];
        if (dijkstraDist != deltaDist) {
            cout << "Mismatch: Node " << node << " -> Dijkstra: " << dijkstraDist 
                 << ", Delta-Stepping: " << deltaDist << "\n";
            isSame = false;
        }
    }

    for (const auto& X : deltaSteppingDistances) {
        auto node = X.first;
        auto deltaDist = X.second;
        if (dijkstraDistances.find(node) == dijkstraDistances.end()) {
            cout << "Mismatch: Node " << node << " is extra in Delta-Stepping output\n";
            isSame = false;
        }
    }

    if (isSame) {
        cout << "Both outputs are identical!\n";
    } else {
        cout << "Outputs differ!\n";
    }

    return 0;
}
