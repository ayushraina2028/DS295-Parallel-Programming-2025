#!/bin/bash

#Take a integer input from the user
echo "Enter the number of processors: "
read n

mpic++ DeltaStepping.cpp -o Executable; mpirun --oversubscribe -np $n Executable