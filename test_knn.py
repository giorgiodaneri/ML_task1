from KNNClassifier import KNNClassifier
from KNNClassifier import KNNClassifierNumba
from KNNClassifierGPU import KNNClassifierGPU
from KNNClassifierDask import KNNClassifierDask
from KNNClassifier import predict_parallel_numba
import numpy as np
import time
import psutil
from joblib import Parallel, delayed
from numba import int32, cuda
import timeit

# Example with random data
rows = 100000
cols = 500
np.random.seed(699)
X_train = np.random.rand(rows*cols).reshape((rows,cols))
y_train = int32(np.random.randint(2, size=rows))
print(f'X_train shape {X_train.shape} - y_train shape {y_train.shape}')

threads = int(psutil.cpu_count(logical=False) / 2)
knn = KNNClassifier(k=2, threads_count=threads)
knn.fit(X_train, y_train)

# Create random indices to test
test_size = 1000
num_runs = 30
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
# print(f'correct {np.sum(y_train[X_test] == predictions)}')

#################### Test Numba implementation ####################
# print('----- Testing Numba implementation -----')
# knn_numba = KNNClassifierGPU(k=2)
# knn_numba.fit(X_train, y_train)
# blocks_per_grid = 2
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
#     print('ERROR: predictions are different')

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

#################### Test Numba implementation ####################
print('----- Testing Numba implementation -----')
threads = int32(threads)
knn_numba = KNNClassifierNumba(k=2, threads_count=threads)
print(f'Using {knn_numba.threads_count} threads')
knn_numba.fit(X_train, y_train)
# start = time.time()
# predictions_numba = predict_parallel_numba(knn_numba, X_train[X_test])
# end = time.time()
# time_numba = end-start
# print(f'Elapsed time for single run of predict_numba {time_numba}')

# now measure time taken by 30 runs of the function with timeit
time_numba = timeit.timeit(lambda: predict_parallel_numba(knn_numba, X_train[X_test]), number=num_runs)
# print average execution time
avg_time = time_numba / num_runs
print(f'Average time for {num_runs} runs of predict_numba: {avg_time}')
print(f'Average speedup using Numba {round((time_single)/(avg_time),2)}')
# compare the output of both methods
# if np.all(predictions == predictions_numba):
#     print('All predictions are exactly equal')
# elif np.allclose(predictions, predictions_numba, rtol=1e-05, atol=1e-08):
#     print('All predictions are close')
# else:
#     print('ERROR: predictions are different')

#################### Test Multiprocessing implementation ####################
print('----- Testing Multiprocessing implementation -----')
print(f'Using {knn.threads_count} threads')
start = time.time()
predictions_mp = knn.predict_multiprocess(X_train[X_test])
end = time.time()
time_mp = end - start
# print the speedup
print(f'Elapsed time for predict_multiprocess {time_mp}')

# now measure time taken by 30 runs of the function with timeit
time_mp = timeit.timeit(lambda: knn.predict_multiprocess(X_train[X_test]), number=num_runs)
# print average execution time
avg_time = time_mp / num_runs
print(f'Average time for {num_runs} runs of predict_multiprocess: {avg_time}')
print(f'Average speedup using multiprocessing {round((time_single)/(avg_time),2)}')
# compare the output of both methods
if np.all(predictions == predictions_mp):
    print('All predictions are exactly equal')
elif np.allclose(predictions, predictions_mp, rtol=1e-05, atol=1e-08):
        print('All predictions are close')
else: 
    print('ERROR: predictions are different')

#################### Test Dask implementation ####################
print('----- Testing Dask implementation -----')
knn_dask = KNNClassifierDask(k=2)
knn_dask.fit(X_train, y_train)
print(f'Using {knn_dask.threads_count} threads')
# start = time.time()
# predictions_dask = knn_dask.predict(X_train[X_test])
# end = time.time()
# time_dask = end-start
# print(f'Elapsed time for single predict_parallel {time_dask}')

# now measure time taken by 30 runs of the function with timeit
time_dask = timeit.timeit(lambda: knn_dask.predict(X_train[X_test]), number=num_runs)
# print average execution time
avg_time = time_dask / num_runs
print(f'Average time for {num_runs} runs of predict_dask: {avg_time}')
print(f'Average speedup using Dask {round((time_single)/(avg_time),2)}')
# compare the output of both methods
# if np.all(predictions == predictions_dask):
#     print('All predictions are exactly equal')
# elif np.allclose(predictions, predictions_dask, rtol=1e-05, atol=1e-08):
#         print('All predictions are close')
# else:
#     print('ERROR: predictions are different')

#################### Test Joblib implementation ####################
print('----- Testing Joblib implementation -----')
print(f'Using {knn.threads_count} threads')
start = time.time()
predictions_joblib = knn.predict_joblib(X_train[X_test])
end = time.time()
time_joblib = end-start
print(f'Elapsed time for predict_joblib {time_joblib}')
# now measure time taken by 30 runs of the function with timeit
time_joblib = timeit.timeit(lambda: knn.predict_joblib(X_train[X_test]), number=num_runs)
# print average execution time
avg_time = time_joblib / num_runs
print(f'Average time for {num_runs} runs of predict_joblib: {avg_time}')
print(f'Average speedup using Joblib {round((time_single)/(avg_time),2)}')
# compare the output of both methods
if np.all(predictions == predictions_joblib):
    print('All predictions are exactly equal')
elif np.allclose(predictions, predictions_joblib, rtol=1e-05, atol=1e-08):
    print('All predictions are close')
else:
    print('ERROR: predictions are different')
