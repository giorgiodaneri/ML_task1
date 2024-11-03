from KNNClassifier import KNNClassifier
from KNNClassifierDask import KNNClassifierDask
import numpy as np
import time

# Example with random data
rows = 10000
cols = 500
np.random.seed(699)
X_train = np.random.rand(rows*cols).reshape((rows,cols))
y_train = np.random.randint(2, size=rows)
print(f'X_train shape {X_train.shape} - y_train shape {y_train.shape}')

knn = KNNClassifier(k=2)
knn.fit(X_train, y_train)

# Create random indices to test
test_size = 100
X_test = np.random.randint(rows, size=test_size)

# Generate Predictions and measure time
start = time.time()
predictions = knn.predict(X_train[X_test])
end = time.time()
time_single = end-start
# time_single = 849.47
print(f'Elapsed time for predict {time_single}')
#print(f'Prediction {predictions}')
#print(f'Label      {y_train[X_test]}')
# Calculate the number of equal elements
# print(f'correct {np.sum(y_train[X_test] == predictions)}')

#################### Test Multiprocessing implementation ####################
start = time.time()
predictions_mp = knn.predict_multiprocess(X_train[X_test])
end = time.time()
time_mp = end - start
# print the speedup
print(f'Elapsed time for predict_multiprocess {time_mp}')
print(f'Speedup using multiprocessing {round((time_single)/(time_mp),2)}')
# Calculate the number of equal elements
print(f'correct {np.sum(y_train[X_test] == predictions_mp)}')
# compare the output of both methods
# if np.all(predictions == predictions_mp):
#     print('All predictions are exactly equal')
# elif np.allclose(predictions, predictions_mp, rtol=1e-05, atol=1e-08):
#         print('All predictions are close')
# else: 
#     print('ERROR: predictions are different')

#################### Test Dask implementation ####################
knn_dask = KNNClassifierDask(k=2)
knn_dask.fit(X_train, y_train)
start = time.time()
predictions_dask = knn_dask.predict(X_train[X_test])
end = time.time()
time_dask = end-start
print(f'Elapsed time for predict_parallel {time_dask}')
print(f'Speedup using Dask {round((time_single)/(time_dask),2)}')
# Calculate the number of equal elements
print(f'correct {np.sum(y_train[X_test] == predictions_dask)}')
# compare the output of both methods
# if np.all(predictions == predictions_dask):
#     print('All predictions are exactly equal')
# elif np.allclose(predictions, predictions_dask, rtol=1e-05, atol=1e-08):
#         print('All predictions are close')

#################### Test Joblib implementation ####################
start = time.time()
predictions_joblib = knn.predict_joblib(X_train[X_test])
end = time.time()
time_joblib = end-start
print(f'Elapsed time for predict_joblib {time_joblib}')
print(f'Speedup using Joblib {round((time_single)/(time_joblib),2)}')
# Calculate the number of equal elements
print(f'correct {np.sum(y_train[X_test] == predictions_joblib)}')
# compare the output of both methods
# if np.all(predictions == predictions_joblib):
#     print('All predictions are exactly equal')
# elif np.allclose(predictions, predictions_joblib, rtol=1e-05, atol=1e-08):
#         print('All predictions are close')