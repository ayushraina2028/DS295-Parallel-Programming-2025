#include <iostream>
#include <fstream>
#include <random>

void generate_random_graph(int n, double edge_prob, const std::string& output_file) {
    std::ofstream fout(output_file);
    if (!fout) {
        std::cerr << "Error opening file: " << output_file << std::endl;
        return;
    }

    // Use high-performance random number generators
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_int_distribution<int> weight_dist(1, 100);

    for (int u = 0; u < n; ++u) {
        for (int v = u + 1; v < n; ++v) {
            if (prob_dist(rng) < edge_prob) {  // Add edge with given probability
                int w = weight_dist(rng);    // Random weight between 1 and 100
                fout << u << " " << v << " " << w << "\n";
            }
        }
    }

    fout.close();
    std::cout << "Graph generated and saved to " << output_file << "\n";
}

int main() {
    int n = 10000;          // Number of vertices
    double edge_prob = 0.1; // Probability of an edge between any two vertices

    generate_random_graph(n, edge_prob, "SyntheticGraph.txt");

    return 0;
}
