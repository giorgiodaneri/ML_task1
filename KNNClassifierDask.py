import numpy as np
import dask.array as da
from dask import delayed, compute
import psutil

class KNNClassifierDask:
    def __init__(self, k=3):
        self.k = k
        self.threads_count = int(psutil.cpu_count(logical=False) / 16)

    def fit(self, X, y):
        self.X_train = da.from_array(X, chunks=(X.shape[0] // self.threads_count, X.shape[1]))
        self.y_train = y

    def euclidean_distance(self, x1, X2):
        diff = x1 - X2
        sqr_diff = diff ** 2
        sqr_diff_sum = da.sum(sqr_diff, axis=1)
        return da.sqrt(sqr_diff_sum)

    def predict(self, X):
        # Convert input data to Dask array
        print(f'Using {self.threads_count} threads')
        X_dask = da.from_array(X, chunks=(X.shape[0] // self.threads_count, X.shape[1]))
        # Use Dask delayed to handle batch prediction
        predictions = [delayed(self._predict)(x) for x in X_dask]
        # Compute the delayed predictions in parallel
        return np.array(compute(*predictions))

    def _predict(self, x):
        # Compute distances from the point x to all training points
        distances = self.euclidean_distance(x, self.X_train).compute()
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Retrieve the labels of these k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common