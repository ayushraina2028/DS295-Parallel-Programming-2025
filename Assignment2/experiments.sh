#!/bin/bash

# Create results directory
mkdir -p results

# Compile the code
mpic++ -o delta_stepping DeltaStepping.cpp -std=c++11

# Run with different processor counts
for p in 1 2 4 8 16; do
    echo "Running with $p processes..."
    mpirun -np $p ./delta_stepping > results/output_$p.txt
    
    # Extract runtime from output
    grep "Maximum time" results_$p.txt >> results/scaling.txt
done

echo "All experiments completed."