#!/bin/bash -l
#SBATCH -J single_proc_run
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --time=0-16:00:00
#SBATCH -p batch

module load mpi/OpenMPI

micromamba activate _your_env_name_

# Start running the sequential simulation
out_file="mpi_res.csv"
touch $out_file
truncate -s 0 $out_file

echo "proc,time" >> $out_file

iterations=30

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
