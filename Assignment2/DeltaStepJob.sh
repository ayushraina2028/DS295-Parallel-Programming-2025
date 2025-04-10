#!/bin/bash
#SBATCH --job-name=delta_stepping      # Job name
#SBATCH --partition=A100               # Use the A100 partition (best for multi-node)
#SBATCH --nodes=2                      # Use 2 nodes for up to 64 processes
#SBATCH --ntasks=64                    # Total MPI processes (32 or 64)
#SBATCH --cpus-per-task=1              # 1 core per MPI process
#SBATCH --time=04:00:00                # Max job time (4 hours)
#SBATCH --output=delta_output_%j.log   # Output log (with job ID for tracking)
#SBATCH --error=delta_error_%j.log     # Error log

# Run with 32 processes first
echo "Running with 32 MPI processes"
mpirun -np 1 ./delta_stepping
