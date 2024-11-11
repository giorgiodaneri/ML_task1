from KNNClassifier import KNNClassifier
import numpy as np
import time
import sys

auto = False

if "-a" in sys.argv or "--auto" in sys.argv:
    auto = True


# Example with random data
rows = 100000
cols = 500
np.random.seed(699)
X_train = np.random.rand(rows*cols).reshape((rows,cols))
y_train = np.random.randint(2, size=rows)


knn = KNNClassifier(k=2)
knn.fit(X_train, y_train)

# Create random indices to test
test_size = 1000
X_test = np.random.randint(rows, size=test_size)

start = time.time()
predictions = knn.predict(X_train[X_test])
end = time.time()

# Generate Predictions

# Calculate the number of equal elements
if not auto:
    print(f'correct {np.sum(y_train[X_test] == predictions)}')
    print(f'Time {end-start}s')
else:
    print(f'{end-start}')