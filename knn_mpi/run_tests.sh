#!/bin/bash

# Start running the sequential simulation
out_file="res.csv"
touch $out_file
truncate -s 0 $out_file

iterations=30

echo "proc,time" >> $out_file

echo "Running parallel simulation"
for procs in 2 4 8 16 32 64 128 256; do
    echo "Running simulation with $procs processes"
    for i in $(seq 1 $iterations); do
        # Run the parallel simulation
        time=$(mpirun -np $procs python ./mpi/test_knn_parallel.py -a)
        echo "Iteration $i -> $time"
        echo "$procs,${time}" >> $out_file;
    done
done

iterations_seq=10
for i in $(seq 1 $iterations_seq); do
    # Run the sequential simulation
    time=$(python ./mpi/test_knn_seq.py -a)
    echo "Iteration $i -> $time"

    echo "1,${time}" >> $out_file;
done
