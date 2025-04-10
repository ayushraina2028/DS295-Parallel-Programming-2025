#!/bin/bash

g++ GenerateGraph.cpp -o Executable; ./Executable
./runDijkstra.sh
./runDeltaStepping.sh