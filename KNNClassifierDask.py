import numpy as np
import dask.array as da
from dask import delayed, compute
import psutil
from multiprocessing import Pool

class KNNClassifierDask:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = da.from_array(X, chunks=(X.shape[0] // psutil.cpu_count(logical=False), X.shape[1]))
        self.y_train = y

    def euclidean_distance(self, x1, X2):
        # Use sqrt and sum functions which are lazily evaluated in parallel
        # by dividing the data into chinks and distributing them across available resources
        diff = x1 - X2
        sqr_diff = diff ** 2
        sqr_diff_sum = da.sum(sqr_diff, axis=1)
        return da.sqrt(sqr_diff_sum)

    def predict(self, X):
        # Convert input data to Dask array
        X_dask = da.from_array(X, chunks=(X.shape[0] // psutil.cpu_count(logical=False), X.shape[1]))
        # Use Dask delayed to handle batch prediction
        predictions = [delayed(self._predict)(x) for x in X_dask]
        # Compute the delayed predictions in parallel
        return np.array(compute(*predictions))
    
    def predict_parallel(self, X):
        # get number of available threads
        threads_count = psutil.cpu_count(logical=False)
        print(f'Using {threads_count} threads')
        # split the data in as many parts as threads
        n = X.shape[0]
        parts = np.array_split(X, threads_count)
        # Create a pool of workers
        pool = Pool(threads_count)
        # Run the prediction in parallel by mapping the function to each part of the input data
        results = pool.map(self.predict, parts)
        # Close the pool
        pool.close()
        pool.join()
        # Concatenate the results
        return np.concatenate(results)

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