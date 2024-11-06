from KNNClassifier import KNNClassifier
from KNNClassifier import KNNClassifierNumba
from KNNClassifierGPU import KNNClassifierGPU
from KNNClassifierDask import KNNClassifierDask
from KNNClassifier import predict_parallel_numba
from joblib import Parallel, delayed
from numba import int32, cuda
import numpy as np
import time
import psutil
import csv
import os

def check_prediction(actual_predictions, test_predictions):
    # if predictions are different return false
    if not np.all(actual_predictions == test_predictions):
        return False
    return True

# Example with random data
rows = 10000
cols = 500
np.random.seed(699)
X_train = np.random.rand(rows*cols).reshape((rows,cols))
y_train = int32(np.random.randint(2, size=rows))
print(f'X_train shape {X_train.shape} - y_train shape {y_train.shape}')

threads = int(psutil.cpu_count(logical=False) / 2)
knn = KNNClassifier(k=2, threads_count=threads)
knn.fit(X_train, y_train)

# Create random indices to test
test_size = 100
num_runs = 30
thread_num = [8, 16, 32, 64]
mp_file_path = "results/mp_times.csv"
dask_file_path = "results/dask_times.csv"
joblib_file_path = "results/joblib_times.csv"
numba_file_path = "results/numba_times.csv"
X_test = np.random.randint(rows, size=test_size)

# Generate Predictions and measure time
start = time.time()
predictions = knn.predict(X_train[X_test])
end = time.time()
time_single = end-start
# time_single = 830
print(f'Elapsed time for predict {time_single}')
#print(f'Prediction {predictions}')
#print(f'Label      {y_train[X_test]}')
# Calculate the number of equal elements
print(f'correct {np.sum(y_train[X_test] == predictions)}')

#################### Test Numba implementation ####################
print('----- Testing Numba implementation -----')
knn_numba = KNNClassifierGPU(k=2)
knn_numba.fit(X_train, y_train)
blocks_per_grid = 2
threads_per_block = 8
start = time.time()
predictions_numba = knn_numba.predict(X_train[X_test], blocks_per_grid, threads_per_block)
end = time.time()
time_numba = end-start
print(f'Elapsed time for GPU predict {time_numba}')
print(f'Average speedup using Numba {round((time_single)/(time_numba),2)}')
# print number of correct predictions
print(f'correct {np.sum(y_train[X_test] == predictions_numba)}')
# compare the output of both methods
if np.all(predictions == predictions_numba):
    print('All predictions are exactly equal')
elif np.allclose(predictions, predictions_numba, rtol=1e-05, atol=1e-08):
    print('All predictions are close')
else:
    print('ERROR: predictions are different')

# blocks_per_grid = 4
# threads_per_block = 8
# start = time.time()
# predictions_numba = knn_numba.predict(X_train[X_test], blocks_per_grid, threads_per_block)
# end = time.time()
# time_numba = end-start
# print(f'Elapsed time for GPU predict {time_numba}')
# print(f'Average speedup using Numba {round((time_single)/(time_numba),2)}')
# # compare the output of both methods
# if np.all(predictions == predictions_numba):
#     print('All predictions are exactly equal')
# elif np.allclose(predictions, predictions_numba, rtol=1e-05, atol=1e-08):
#     print('All predictions are close')
# else:
    # print('ERROR: predictions are different')

# blocks_per_grid = 8
# threads_per_block = 8
# start = time.time()
# predictions_numba = knn_numba.predict(X_train[X_test], blocks_per_grid, threads_per_block)
# end = time.time()
# time_numba = end-start
# print(f'Elapsed time for GPU predict {time_numba}')
# print(f'Average speedup using Numba {round((time_single)/(time_numba),2)}')
# # compare the output of both methods
# if np.all(predictions == predictions_numba):
#     print('All predictions are exactly equal')
# elif np.allclose(predictions, predictions_numba, rtol=1e-05, atol=1e-08):
#     print('All predictions are close')
# else:
#     print('ERROR: predictions are different')

# blocks_per_grid = 8
# threads_per_block = 16
# start = time.time()
# predictions_numba = knn_numba.predict(X_train[X_test], blocks_per_grid, threads_per_block)
# end = time.time()
# time_numba = end-start
# print(f'Elapsed time for GPU predict {time_numba}')
# print(f'Average speedup using Numba {round((time_single)/(time_numba),2)}')
# # compare the output of both methods
# if np.all(predictions == predictions_numba):
#     print('All predictions are exactly equal')
# elif np.allclose(predictions, predictions_numba, rtol=1e-05, atol=1e-08):
#     print('All predictions are close')
# else:
#     print('ERROR: predictions are different')

# #################### Test Multiprocessing implementation ####################
# print('----- Testing Multiprocessing implementation -----')
# file_exists = os.path.isfile(mp_file_path)
# with open(mp_file_path, mode="a", newline="") as csv_file:
#     writer = csv.writer(csv_file)

#     if not file_exists:
#         writer.writerow(["Threads", "Time"])

#     # iterate over different number of threads
#     for threads in thread_num:
#         knn = KNNClassifier(k=2, threads_count=threads)
#         print(f'Using {knn.threads_count} threads')
#         knn.fit(X_train, y_train)

#         predictions_mp = np.zeros(test_size)
#         for i in range(num_runs):
#             start = time.time()
#             predictions_mp = knn.predict_multiprocess(X_train[X_test])
#             end = time.time()
#             mp_time = end-start
#             print(f'correct {np.sum(y_train[X_test] == predictions_mp)}')
#             writer.writerow([threads, mp_time])

# #################### Test Numba implementation ####################
# print('----- Testing Numba implementation -----')
# # Check if the file exists to determine if we need to write headers
# file_exists = os.path.isfile(numba_file_path)
# with open(numba_file_path, mode="a", newline="") as csv_file:
#     writer = csv.writer(csv_file)

#     if not file_exists:
#         writer.writerow(["Threads", "Time"])

#     # iterate over different number of threads
#     for threads in thread_num:
#         threads = int32(threads)
#         knn_numba = KNNClassifierNumba(k=2, threads_count=threads)
#         print(f'Using {knn_numba.threads_count} threads')
#         knn_numba.fit(X_train, y_train)

#         predictions_numba = np.zeros(test_size)
#         for i in range(num_runs):
#             start = time.time()
#             predictions_numba = predict_parallel_numba(knn_numba, X_train[X_test])
#             end = time.time()
#             numba_time = end-start
#             print(f'correct {np.sum(y_train[X_test] == predictions_numba)}')
#             writer.writerow([threads, numba_time])   

# #################### Test Dask implementation ####################
# print('----- Testing Dask implementation -----')
# file_exists = os.path.isfile(dask_file_path)
# with open(dask_file_path, mode="a", newline="") as csv_file:
#     writer = csv.writer(csv_file)

#     if not file_exists:
#         writer.writerow(["Threads", "Time"])

#     # iterate over different number of threads
#     for threads in thread_num:
#         knn_dask = KNNClassifierDask(k=2, threads_count=threads)
#         print(f'Using {knn_dask.threads_count} threads')
#         knn_dask.fit(X_train, y_train)

#         predictions_dask = np.zeros(test_size)
#         for i in range(num_runs):
#             start = time.time()
#             predictions_dask = knn_dask.predict(X_train[X_test])
#             end = time.time()
#             dask_time = end-start
#             print(f'correct {np.sum(y_train[X_test] == predictions_dask)}')
#             writer.writerow([threads, dask_time])

# #################### Test Joblib implementation ####################
# print('----- Testing Joblib implementation -----')
# file_exists = os.path.isfile(joblib_file_path)
# with open(joblib_file_path, mode="a", newline="") as csv_file:
#     writer = csv.writer(csv_file)

#     if not file_exists:
#         writer.writerow(["Threads", "Time"])

#     # iterate over different number of threads
#     for threads in thread_num:
#         knn = KNNClassifier(k=2, threads_count=threads)
#         print(f'Using {knn.threads_count} threads')
#         knn.fit(X_train, y_train)

#         predictions_joblib = np.zeros(test_size)
#         for i in range(num_runs):
#             start = time.time()
#             predictions_joblib = knn.predict_joblib(X_train[X_test])
#             end = time.time()
#             joblib_time = end-start
#             print(f'correct {np.sum(y_train[X_test] == predictions_joblib)}')
#             writer.writerow([threads, joblib_time])