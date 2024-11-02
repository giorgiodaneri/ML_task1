#!/bin/bash -l              
### Request a single task using one core on one node for 5 minutes in the batch queue
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH --time=0-00:30:00
#SBATCH -p batch
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --job-name="ML_task1"

# activate the virtual environment
source /home/users/gdaneri/.venv/bin/activate

# Run the python script and redirect the output to a file
python3 /home/users/gdaneri/ML_task1/test_knn.py
