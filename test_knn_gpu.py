from KNNClassifier import KNNClassifier
from KNNClassifierGPU import KNNClassifierGPU
from numba import int32, cuda
import numpy as np
import time

# Example with random data
rows = 10000
cols = 500
np.random.seed(699)
X_train = np.random.rand(rows*cols).reshape((rows,cols))
y_train = int32(np.random.randint(2, size=rows))
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
