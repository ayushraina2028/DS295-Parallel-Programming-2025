#include <mpi.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <string>
#include <sstream>
#include <algorithm>

void printAdjacencyList(const std::unordered_map<int, std::set<int>>& adj_list, int rank) {
    std::cout << "Processor " << rank << " adjacency list:" << std::endl;
    for (const auto& pair : adj_list) {
        std::cout << pair.first << ": ";
        for (int neighbor : pair.second) {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl;
    }
}

class ParallelEdgeReader {
private:
    MPI_File file;
    int rank, size;
    MPI_Offset filesize;

    // Helper function to read a line
    bool readLine(char* buffer, MPI_Offset& current_pos, int& src, int& dest) {
        char line[100];  // Adjust buffer size as needed
        int line_index = 0;
        
        // Read characters until newline or end of buffer
        while (current_pos < filesize) {
            char ch;
            MPI_File_read_at(file, current_pos, &ch, 1, MPI_CHAR, MPI_STATUS_IGNORE);
            current_pos++;

            if (ch == '\n') break;
            line[line_index++] = ch;
        }
        line[line_index] = '\0';

        // Parse the line
        std::istringstream iss(line);
        return (bool)(iss >> src >> dest);
    }

public:
    ParallelEdgeReader(const char* filename, MPI_Comm comm) {
        // Get process rank and total number of processes
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        // Open the file
        int rc = MPI_File_open(comm, filename, 
                               MPI_MODE_RDONLY, 
                               MPI_INFO_NULL, 
                               &file);
        
        if (rc != MPI_SUCCESS) {
            throw std::runtime_error("Error opening file");
        }

        // Get file size
        MPI_File_get_size(file, &filesize);
    }

    std::unordered_map<int, std::set<int>> readEdges() {
        std::unordered_map<int, std::set<int>> local_adj_list;
        
        // Calculate start and end positions for this process
        MPI_Offset chunk_size = filesize / size;
        MPI_Offset start = rank * chunk_size;
        MPI_Offset end = (rank == size - 1) ? filesize : start + chunk_size;

        // Adjust start position to beginning of a line
        if (rank != 0) {
            // Move to the first full line
            char ch;
            while (start < end) {
                MPI_File_read_at(file, start, &ch, 1, MPI_CHAR, MPI_STATUS_IGNORE);
                start++;
                if (ch == '\n') break;
            }
        }

        // Read edges for this process
        MPI_Offset current_pos = start;
        while (current_pos < end) {
            int src, dest;
            if (readLine(nullptr, current_pos, src, dest)) {
                local_adj_list[src].insert(dest);
                local_adj_list[dest].insert(src);
            }
        }

        return local_adj_list;
    }

    ~ParallelEdgeReader() {
        MPI_File_close(&file);
    }
};

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    try {
        // Create parallel edge reader
        ParallelEdgeReader reader("edges.txt", MPI_COMM_WORLD);

        // Read edges
        auto local_adj_list = reader.readEdges();

        // Get process rank
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Print local adjacency list
        printAdjacencyList(local_adj_list, rank);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}