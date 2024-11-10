#!/bin/bash -l              
### Request a single task using one core on one node for 5 minutes in the batch queue
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --time=0-06:00:00
#SBATCH -p batch
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --job-name="ML_task1"
# Load the required modules
module purge
module load lang/Python/3.8.6-GCCcore-10.2.0
# activate the virtual environment
source /path/to/your/virtual/env/bin/activate
# Run the python script and redirect the output to a file
# change the name of the python script to the one you want to run
python3 test_knn_numba.py
# python3 test_knn_dask.py
# python3 test_knn_mp.py
# python3 test_knn_joblib.py