from KNNClassifier import KNNClassifier
from KNNClassifier import KNNClassifierNumba
from KNNClassifier import predict_parallel_numba
from numba import int32
import numpy as np
import time
import psutil
import csv
import os

print('----- Testing Numba implementation -----')
# Example with random data
rows = 100000
cols = 500
np.random.seed(699)
X_train = np.random.rand(rows*cols).reshape((rows,cols))
y_train = int32(np.random.randint(2, size=rows))
print(f'X_train shape {X_train.shape} - y_train shape {y_train.shape}')
# Create random indices to test
test_size = 1000

# ---------------------- Simulation parameters ---------------------- #
num_runs = 30
thread_num = [8, 16, 32, 64]

numba_file_path = "results/numba_times.csv"
X_test = np.random.randint(rows, size=test_size)

# Check if the file exists to determine if we need to write headers
file_exists = os.path.isfile(numba_file_path)
with open(numba_file_path, mode="a", newline="") as csv_file:
    writer = csv.writer(csv_file)

    if not file_exists:
        writer.writerow(["Threads", "Time"])

    # iterate over different number of threads
    for threads in thread_num:
        threads = int32(threads)
        knn_numba = KNNClassifierNumba(k=2, threads_count=threads)
        print(f'Using {knn_numba.threads_count} threads')
        knn_numba.fit(X_train, y_train)

        predictions_numba = np.zeros(test_size)
        for i in range(num_runs):
            start = time.time()
            predictions_numba = predict_parallel_numba(knn_numba, X_train[X_test])
            end = time.time()
            numba_time = end-start
            print(f'correct {np.sum(y_train[X_test] == predictions_numba)}')
            writer.writerow([threads, numba_time])   