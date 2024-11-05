#!/bin/bash -l
### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH -G 1
#SBATCH --time=02:00:00
#SBATCH -p gpu
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --job-name="ML_task1"

# Load the required modules
module purge
module load lang/Python/3.8.6-GCCcore-10.2.0
module load toolchain/fosscuda
# activate the virtual environment
source /home/users/gdaneri/ML_task1/.dask_venv/bin/activate
# Run the python script and redirect the output to a file
python3 /home/users/gdaneri/ML_task1/test_knn.py