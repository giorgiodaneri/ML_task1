from KNNClassifier import ParallelKNNClassifier
import numpy as np
from mpi4py import MPI
import time
import sys

auto = False

if "-a" in sys.argv or "--auto" in sys.argv:
    auto = True

start = time.time()
if not MPI.Is_initialized():
    MPI.Init()

knn = ParallelKNNClassifier(k=2)
if knn.rank == 0:
    # Example with random data
    rows = 100000
    cols = 500
    np.random.seed(699)
    X_train = np.random.rand(rows*cols).reshape((rows,cols))
    y_train = np.random.randint(2, size=rows)
else:
    X_train = None
    y_train = None


knn.fit(X_train, y_train)

# Create random indices to test
test_size = 1000

if knn.rank == 0:
    X_test = np.random.randint(rows, size=test_size)
else:
    X_test = None

if knn.rank == 0:
    predictions = knn.predict(X_train[X_test])
else:
    predictions = knn.predict(None)
    
end = time.time()

# Generate Predictions
# Calculate the number of equal elements
if knn.rank == 0 and not auto:
    print(f'correct {np.sum(y_train[X_test] == predictions)}')
    print(f'Time {end-start}s')

if knn.rank == 0 and auto:
    print(f'{end-start}')

if not MPI.Is_finalized():
    MPI.Finalize()
