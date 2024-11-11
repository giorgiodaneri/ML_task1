from KNNClassifierDask import KNNClassifierDask
import numpy as np
import time
import csv
import os

print('----- Testing Dask implementation -----')
# Example with random data
rows = 100000
cols = 500
np.random.seed(699)
X_train = np.random.rand(rows*cols).reshape((rows,cols))
y_train = np.random.randint(2, size=rows)
print(f'X_train shape {X_train.shape} - y_train shape {y_train.shape}')
# Create random indices to test
test_size = 1000

# ---------------------- Simulation parameters ---------------------- #
num_runs = 30
thread_num = [8, 16, 32, 64]

dask_file_path = "../results/dask_times.csv"
X_test = np.random.randint(rows, size=test_size)
file_exists = os.path.isfile(dask_file_path)
with open(dask_file_path, mode="a", newline="") as csv_file:
    writer = csv.writer(csv_file)

    if not file_exists:
        writer.writerow(["Threads", "Time"])

    # iterate over different number of threads
    for threads in thread_num:
        knn_dask = KNNClassifierDask(k=2, threads_count=threads)
        print(f'Using {knn_dask.threads_count} threads')
        knn_dask.fit(X_train, y_train)

        predictions_dask = np.zeros(test_size)
        for i in range(num_runs):
            start = time.time()
            predictions_dask = knn_dask.predict(X_train[X_test])
            end = time.time()
            dask_time = end-start
            print(f'correct {np.sum(y_train[X_test] == predictions_dask)}')
            writer.writerow([threads, dask_time])