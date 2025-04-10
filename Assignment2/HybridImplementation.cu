#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <set>
#include <unordered_set>
#include <limits>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <chrono>

// Use a specific constant for infinity that works on both host and device
#define DIST_INF 2147483647  // Same as INT_MAX but explicitly defined

using namespace std;
using namespace std::chrono;

// Fixed relaxEdgesKernel implementation
__global__ void relaxEdgesKernel(
    int* d_edges_src, int* d_edges_dst, int* d_edges_weight, int num_edges,
    int* d_dist, bool* d_changed, int* d_bucket_nodes, int num_bucket_nodes,
    int delta, int current_bucket) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_bucket_nodes) {
        int node = d_bucket_nodes[idx];
        
        // Only process if the node's distance is valid
        if (d_dist[node] != DIST_INF) {
            // Iterate through edges
            for (int e = 0; e < num_edges; e++) {
                if (d_edges_src[e] == node) {
                    int v = d_edges_dst[e];
                    int w = d_edges_weight[e];
                    
                    // Calculate new distance
                    int new_dist = d_dist[node] + w;
                    
                    // Check for overflow and only update if new_dist is less
                    if (new_dist >= 0 && (d_dist[v] == DIST_INF || new_dist < d_dist[v])) {
                        atomicMin(&d_dist[v], new_dist);
                        *d_changed = true;
                        // Debug - but can't print from GPU kernel
                        // printf("Updated node %d distance to %d\n", v, new_dist);
                    }
                }
            }
        }
    }
}

// Rest of your code remains the same...

// Check CUDA errors
void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(error) << endl;
        exit(EXIT_FAILURE);
    }
}

// Graph data structure
struct Graph {
    int num_nodes = 0;
    int num_edges = 0;
    vector<vector<pair<int, int>>> adj_list; // [node] -> list of (dest, weight)
    vector<int> src, dst, weight; // Edge list for CUDA
};

// Read graph from file
Graph readGraph(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(EXIT_FAILURE);
    }
    
    Graph graph;
    int u, v, w;
    
    // First pass - count unique nodes
    while (file >> u >> v >> w) {
        graph.num_nodes = max(graph.num_nodes, max(u, v) + 1);
    }
    
    // Initialize adjacency list
    graph.adj_list.resize(graph.num_nodes);
    
    // Reset file pointer
    file.clear();
    file.seekg(0);
    
    // Second pass - add all edges (both directions for undirected graph)
    while (file >> u >> v >> w) {
        // Original direction
        graph.src.push_back(u);
        graph.dst.push_back(v);
        graph.weight.push_back(w);
        graph.adj_list[u].push_back({v, w});
        
        // Reverse direction for undirected graph
        // graph.src.push_back(v);
        // graph.dst.push_back(u);
        // graph.weight.push_back(w);
        // graph.adj_list[v].push_back({u, w});
        
        graph.num_edges += 1; // Count both directions
    }
    
    file.close();
    cout << "Graph loaded with " << graph.num_nodes << " nodes and " 
         << graph.num_edges << " edges (including both directions)." << endl;
    
    return graph;
}

// Struct to hold performance metrics
struct PerformanceMetrics {
    double total_time_seconds;
    int iteration_count;
    vector<double> per_bucket_time;
    vector<int> bucket_sizes;
    double data_transfer_time;
    double kernel_execution_time;
    double bucket_processing_time;
};

// Main delta-stepping algorithm with hybrid OpenMP-CUDA implementation
vector<int> deltaSteppingHybrid(const Graph& graph, int source, int delta, PerformanceMetrics& metrics) {
    int num_nodes = graph.num_nodes;
    int num_edges = graph.num_edges;
    
    // Initialize distance array
    vector<int> dist(num_nodes, DIST_INF);
    dist[source] = 0;   
    
    // Initialize buckets - each bucket contains nodes whose tentative distance
    // falls within a particular range - use more buckets than nodes
    const int MAX_BUCKET_COUNT = 1000; // Much larger than needed, but safe
    vector<set<int>> buckets(MAX_BUCKET_COUNT);
    buckets[0].insert(source);
    
    // Start timing for data transfer
    auto data_transfer_start = high_resolution_clock::now();
    
    // Transfer graph data to device
    int *d_edges_src, *d_edges_dst, *d_edges_weight, *d_dist;
    bool *d_changed;
    int *d_bucket_nodes;
    
    // Allocate device memory
    checkCudaError(cudaMalloc(&d_edges_src, num_edges * sizeof(int)), "allocate d_edges_src");
    checkCudaError(cudaMalloc(&d_edges_dst, num_edges * sizeof(int)), "allocate d_edges_dst");
    checkCudaError(cudaMalloc(&d_edges_weight, num_edges * sizeof(int)), "allocate d_edges_weight");
    checkCudaError(cudaMalloc(&d_dist, num_nodes * sizeof(int)), "allocate d_dist");
    checkCudaError(cudaMalloc(&d_changed, sizeof(bool)), "allocate d_changed");
    checkCudaError(cudaMalloc(&d_bucket_nodes, num_nodes * sizeof(int)), "allocate d_bucket_nodes");
    
    // Copy edge data to device (these don't change)
    checkCudaError(cudaMemcpy(d_edges_src, graph.src.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice), "copy edges_src");
    checkCudaError(cudaMemcpy(d_edges_dst, graph.dst.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice), "copy edges_dst");
    checkCudaError(cudaMemcpy(d_edges_weight, graph.weight.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice), "copy edges_weight");
    
    // Copy initial distance array to device
    checkCudaError(cudaMemcpy(d_dist, dist.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice), "copy initial dist");
    
    auto data_transfer_end = high_resolution_clock::now();
    metrics.data_transfer_time = duration_cast<microseconds>(data_transfer_end - data_transfer_start).count() / 1000000.0;
    
    // Process buckets
    int current_bucket = 0;
    bool global_changed = true;
    int iteration_count = 0;
    const int MAX_ITERATIONS = 1000; // Safety mechanism
    double total_kernel_time = 0.0;
    double total_bucket_processing_time = 0.0;

    while (global_changed && iteration_count < MAX_ITERATIONS) {
        iteration_count++;
        global_changed = false;
        
        // Print progress every few iterations for debugging
        if (iteration_count % 10 == 0) {
            cout << "Processing iteration " << iteration_count << endl;
        }
        
        // Find the smallest non-empty bucket
        int smallest_bucket = -1;
        for (int b = current_bucket; b < buckets.size(); b++) {
            if (!buckets[b].empty()) {
                smallest_bucket = b;
                break;
            }
        }
        
        // Exit if no more non-empty buckets
        if (smallest_bucket == -1) break;
        
        current_bucket = smallest_bucket;
        
        // Process the current bucket
        vector<int> bucket_nodes(buckets[smallest_bucket].begin(), buckets[smallest_bucket].end());
        buckets[smallest_bucket].clear();
        
        // Record bucket size for metrics
        metrics.bucket_sizes.push_back(bucket_nodes.size());
        
        auto bucket_start_time = high_resolution_clock::now();
        
        // Reset changed flag
        bool changed = false;
        checkCudaError(cudaMemcpy(d_changed, &changed, sizeof(bool), cudaMemcpyHostToDevice), "reset changed flag");
        
        // Copy bucket nodes to device
        checkCudaError(cudaMemcpy(d_bucket_nodes, bucket_nodes.data(), bucket_nodes.size() * sizeof(int), 
                      cudaMemcpyHostToDevice), "copy bucket nodes");
        
        // Launch kernel to relax edges
        int threadsPerBlock = 256;
        int blocksPerGrid = (bucket_nodes.size() + threadsPerBlock - 1) / threadsPerBlock;
        
        auto kernel_start = high_resolution_clock::now();
        
        relaxEdgesKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_edges_src, d_edges_dst, d_edges_weight, num_edges,
            d_dist, d_changed, d_bucket_nodes, bucket_nodes.size(),
            delta, smallest_bucket
        );
        
        // Check for kernel errors
        checkCudaError(cudaGetLastError(), "launching kernel");
        checkCudaError(cudaDeviceSynchronize(), "kernel execution");
        
        auto kernel_end = high_resolution_clock::now();
        double kernel_time = duration_cast<microseconds>(kernel_end - kernel_start).count() / 1000000.0;
        total_kernel_time += kernel_time;
        
        // Copy changed flag and distances back to host
        checkCudaError(cudaMemcpy(&changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost), "copy changed flag");
        checkCudaError(cudaMemcpy(dist.data(), d_dist, num_nodes * sizeof(int), cudaMemcpyDeviceToHost), "copy distances");
        
        // Update global_changed flag
        global_changed = changed;
        
        // In deltaSteppingHybrid function, verify this section:
        if (changed) {
            cout << "Changes detected, reassigning nodes to buckets..." << endl;
            for (int i = 0; i < num_nodes; i++) {
                // If distance was updated and is now finite
                if (dist[i] != DIST_INF) {  // Use DIST_INF consistently
                    int bucket_id = dist[i] / delta;
                    
                    // Safety check - should not happen with MAX_BUCKET_COUNT
                    if (bucket_id >= buckets.size()) {
                        cerr << "Error: Bucket ID " << bucket_id << " out of range!" << endl;
                        continue;
                    }
                    
                    // Only insert if not already processed in current iteration
                    if (i != source && find(bucket_nodes.begin(), bucket_nodes.end(), i) == bucket_nodes.end()) {
                        // cout << "Node " << i << " (dist=" << dist[i] << ") assigned to bucket " << bucket_id << endl;
                        buckets[bucket_id].insert(i);
                    }
                }
            }
        }
        
        auto bucket_end_time = high_resolution_clock::now();
        double bucket_time = duration_cast<microseconds>(bucket_end_time - bucket_start_time).count() / 1000000.0;
        metrics.per_bucket_time.push_back(bucket_time);
        total_bucket_processing_time += bucket_time;
    }

    if (iteration_count >= MAX_ITERATIONS) {
        cout << "WARNING: Algorithm stopped after reaching maximum iterations!" << endl;
    }
    
    // Update performance metrics
    metrics.iteration_count = iteration_count;
    metrics.kernel_execution_time = total_kernel_time;
    metrics.bucket_processing_time = total_bucket_processing_time;
    
    // Free device memory
    cudaFree(d_edges_src);
    cudaFree(d_edges_dst);
    cudaFree(d_edges_weight);
    cudaFree(d_dist);
    cudaFree(d_changed);
    cudaFree(d_bucket_nodes);
    
    return dist;
}

bool compareOutputFiles(const string& dijkstraOutput, const string& deltaSteppingOutput) {
    ifstream dijkstraOutputFile(dijkstraOutput);
    ifstream deltaSteppingOutputFile(deltaSteppingOutput);

    string dijkstraLine, deltaSteppingLine;

    // Skip the first line of both files
    getline(dijkstraOutputFile, dijkstraLine);
    getline(deltaSteppingOutputFile, deltaSteppingLine);    

    // Compare the rest of the lines
    bool same = true;
    int lineNumber = 1;
    while(getline(dijkstraOutputFile, dijkstraLine) && getline(deltaSteppingOutputFile, deltaSteppingLine)) {

        if(dijkstraLine != deltaSteppingLine) {
            same = false;
            cout << "Different output at line " << lineNumber << ":\n";
            cout << "Dijkstra: " << dijkstraLine << "\n";
            cout << "Delta-Stepping: " << deltaSteppingLine << "\n";
            break;
        }


        lineNumber++;
    }

    if(!same) {
        return false;
    }

    if(dijkstraOutputFile.eof() != deltaSteppingOutputFile.eof()) {
        cout << "Both files have different number of lines\n";
        cout << "Dijkstra File EOF: " << dijkstraOutputFile.eof() << "\n";
        cout << "Delta-Stepping File EOF: " << deltaSteppingOutputFile.eof() << "\n";
        return false;
    } 

    return true;
}

int main(int argc, char* argv[]) {

    if (argc != 3) {  // Expecting two integers
        std::cerr << "Usage: " << argv[0] << " <integer1> <integer2>\n";
        return 1;  // Exit with error
    }

    // Convert command-line arguments to integers
    int a = std::atoi(argv[1]);
    int b = std::atoi(argv[2]);

    // Set OpenMP threads
    omp_set_num_threads(8);
    
    // Read graph - use our test graph
    Graph graph = readGraph("SyntheticGraph.txt");
    
    // Define source and delta parameter
    int source = 0;  // Starting node
    int delta = b;   // Bucket width set to match the smallest edge weight
    
    cout << "Starting hybrid delta-stepping with source node " << source << " and delta " << delta << endl;
    
    // Performance metrics struct
    PerformanceMetrics metrics;
    
    // Measure performance
    auto total_start_time = high_resolution_clock::now();
    
    // Run algorithm
    vector<int> distances = deltaSteppingHybrid(graph, source, delta, metrics);
    
    auto total_end_time = high_resolution_clock::now();
    metrics.total_time_seconds = duration_cast<microseconds>(total_end_time - total_start_time).count() / 1000000.0;
    
    cout << "Algorithm completed in " << metrics.total_time_seconds << " seconds" << endl;
    
    // Print results
    cout << "Shortest paths from node " << source << ":" << endl;
    for (int i = 0; i < min(graph.num_nodes,10); i++) {
        if (distances[i] == DIST_INF) {
            cout << "To node " << i << ": INF" << endl;
        } else {
            cout << "To node " << i << ": " << distances[i] << endl;
        }
    }
    
    // Write results
    ofstream outfile("hybridDeltaSteppingResults.txt");
    if (!outfile.is_open()) {
        cerr << "Error opening output file" << endl;
        return 1;
    }
    
    outfile << "Shortest Paths from Source Vertex " << source << " using Hybrid Delta-Stepping" << endl;
    for (int i = 0; i < graph.num_nodes; i++) {
        if (distances[i] == DIST_INF) {
            outfile << "To " << i << ": INF";
        } else {
            outfile << "To " << i << ": " << distances[i];
        }

        if(i != graph.num_nodes - 1) {
            outfile << endl;
        }
    }
    
    outfile.close();
    
    // Write performance metrics to file
    string performanceFilename = "performance_metrics_" + to_string(a) + ".txt";
    cout << "Writing performance metrics to " << performanceFilename << endl;
    ofstream perfFile(performanceFilename);
    if (!perfFile.is_open()) {
        cerr << "Error opening performance metrics file" << endl;
        return 1;
    }
    
    // Predefined sequential time
    double sequential_time = 4.50; // Seconds

    // Get number of available threads/cores
    int num_threads = omp_get_max_threads();

    // Calculate speedup
    double speedup = sequential_time / metrics.total_time_seconds;

    // Calculate efficiency
    double efficiency = speedup / num_threads;

    perfFile << "DELTA-STEPPING PERFORMANCE METRICS" << endl;
    perfFile << "=================================" << endl;
    perfFile << "Graph Information:" << endl;
    perfFile << "  Nodes: " << graph.num_nodes << endl;
    perfFile << "  Edges: " << graph.num_edges << endl;
    perfFile << "  Source Node: " << source << endl;
    perfFile << "  Delta Value: " << delta << endl << endl;
    
    perfFile << "Timing Information:" << endl;
    perfFile << "  Total Execution Time: " << metrics.total_time_seconds << " seconds" << endl;
    perfFile << "  Initial Data Transfer Time: " << metrics.data_transfer_time << " seconds" << endl;
    perfFile << "  Total Kernel Execution Time: " << metrics.kernel_execution_time << " seconds" << endl;
    perfFile << "  Total Bucket Processing Time: " << metrics.bucket_processing_time << " seconds" << endl << endl;
    
    perfFile << "Parallel Performance Metrics:" << endl;
    perfFile << "  Number of Threads: " << num_threads << endl;
    perfFile << "  Sequential Execution Time: " << sequential_time << " seconds" << endl;
    perfFile << "  Parallel Execution Time: " << metrics.total_time_seconds << " seconds" << endl;
    perfFile << "  Speedup: " << speedup << "x" << endl;
    perfFile << "  Efficiency: " << (efficiency * 100) << "%" << endl << endl;
    
    perfFile << "Algorithm Statistics:" << endl;
    perfFile << "  Iterations Completed: " << metrics.iteration_count << endl;
    perfFile << "  Number of Buckets Processed: " << metrics.bucket_sizes.size() << endl;
    
    // Additional metrics
    double compression_factor = 1.0 * graph.num_nodes / metrics.iteration_count;
    double edge_relaxation_rate = 1.0 * graph.num_edges / metrics.total_time_seconds;
    double computational_intensity = 1.0 * graph.num_edges / metrics.kernel_execution_time;

    perfFile << endl << "Algorithmic Efficiency Metrics:" << endl;
    perfFile << "  Compression Factor: " << compression_factor << endl;
    perfFile << "  Edge Relaxation Rate: " << edge_relaxation_rate << " edges/second" << endl;
    perfFile << "  Computational Intensity: " << computational_intensity << " edges/second/kernel" << endl;
    
    perfFile << endl << "Detailed Per-Bucket Performance:" << endl;
    perfFile << "  Bucket ID | Size | Processing Time (seconds)" << endl;
    perfFile << "  -----------------------------------------" << endl;
    for (size_t i = 0; i < metrics.bucket_sizes.size(); i++) {
        perfFile << "  " << i << " | " << metrics.bucket_sizes[i] << " | " << metrics.per_bucket_time[i] << endl;
    }
    
    perfFile.close();
    cout << "Performance metrics written to performance_metrics.txt" << endl;
    
    bool isResultMatch = compareOutputFiles("dijkstra_output.txt", "HybridDeltaSteppingResults.txt");
    if(isResultMatch) {
        cout << "MATCH ✅\n";
    }
    else {
        cout << "NO MATCH ❌\n";
    }
    
    cout << "Results written to hybridDeltaSteppingResults.txt" << endl;
    
    return 0;
}