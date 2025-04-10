#!/bin/bash
nvcc -Xcompiler -fopenmp HybridImplementation.cu -o Executable; 
./Executable 1 1
./Executable 2 5
./Executable 3 10
./Executable 4 20
./Executable 5 30
./Executable 6 40
./Executable 7 50
./Executable 8 60
./Executable 9 70
./Executable 10 80
./Executable 11 90
./Executable 12 100
