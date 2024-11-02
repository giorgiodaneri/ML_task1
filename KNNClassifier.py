import numpy as np
import dask.array as da
from multiprocessing import Pool
import psutil
from joblib import Parallel,delayed 

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.threads_count = int(psutil.cpu_count(logical=False) / 2)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        diff = (x1 - x2)
        sqr_diff = diff ** 2
        sqr_diff_sum = np.sum(sqr_diff)
        return np.sqrt(sqr_diff_sum)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def predict_joblib(self, X):
        print(f'Using {self.threads_count} threads')
        # run the prediction in parallel using joblib
        y_pred = Parallel(n_jobs=self.threads_count)(delayed(self._predict)(x) for x in X)
        # y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def predict_multiprocess(self, X):
        print(f'Using {self.threads_count} threads')
        # split the data in as many parts as threads
        n = X.shape[0]
        parts = np.array_split(X, self.threads_count)
        # Create a pool of workers
        pool = Pool(self.threads_count)
        # Run the prediction in parallel by mapping the function to each part of the input data
        results = pool.map(self.predict, parts)
        # Close the pool
        pool.close()
        pool.join()
        # Concatenate the results
        return np.concatenate(results)

    def _predict(self, x):
        # Calculate distances from the input point to all training points
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label among the k nearest neighbors
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common