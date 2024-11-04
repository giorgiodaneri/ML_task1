import numpy as np
import psutil
from multiprocessing import Pool
from joblib import Parallel, delayed
from line_profiler import profile
from numba import float64, int32
from numba.experimental import jitclass

class KNNClassifier:
    def __init__(self, k=3, threads_count=8):
        self.k = k
        self.threads_count = threads_count
        self.X_train = np.empty((0, 0), dtype=np.float64)  # Initialize empty array
        self.y_train = np.empty(0, dtype=np.int32)         # Initialize empty array

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        diff = (x1 - x2)
        sqr_diff = diff ** 2
        sqr_diff_sum = np.sum(sqr_diff)
        return np.sqrt(sqr_diff_sum)

    @profile
    def predict(self, X):
        y_pred = np.empty(X.shape[0], dtype=np.int32)
        for i in range(X.shape[0]):
            y_pred[i] = self._predict(X[i])
        return y_pred
    
    @profile
    def predict_joblib(self, X):
        print(f'Using {self.threads_count} threads')
        y_pred = Parallel(n_jobs=self.threads_count)(delayed(self._predict)(x) for x in X)
        return np.array(y_pred)

    # Define a separate function for multiprocessing
    def predict_multiprocess(knn_classifier, X):
        print(f'Using {knn_classifier.threads_count} threads')
        n = X.shape[0]
        parts = np.array_split(X, knn_classifier.threads_count)
        results = []

        # Create a pool of workers
        pool = Pool(knn_classifier.threads_count)
        results = pool.map(knn_classifier.predict, parts)
        pool.close()
        pool.join()
        
        # Concatenate the results
        return np.concatenate(results)
    
    def _predict(self, x):
        # Calculate distances from the input point to all training points
        distances = np.empty(self.X_train.shape[0])
        for i in range(self.X_train.shape[0]):
            distances[i] = self.euclidean_distance(x, self.X_train[i])
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = self.y_train[k_indices]
        # Return the most common class label among the k nearest neighbors
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
    

# Define the specification for the KNNClassifier
spec = [
    ('k', int32),                # Number of neighbors
    ('threads_count', int32),    # Number of threads
    ('X_train', float64[:, :]),  # 2D NumPy array for training data
    ('y_train', int32[:]),        # 1D NumPy array for training labels
]

@jitclass(spec)
class KNNClassifierNumba:
    def __init__(self, k=3, threads_count=8):
        self.k = k
        self.threads_count = threads_count
        self.X_train = np.empty((0, 0), dtype=np.float64)  # Initialize empty array
        self.y_train = np.empty(0, dtype=np.int32)         # Initialize empty array

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        diff = (x1 - x2)
        sqr_diff = diff ** 2
        sqr_diff_sum = np.sum(sqr_diff)
        return np.sqrt(sqr_diff_sum)

    @profile
    def predict(self, X):
        print(f'Using {self.threads_count} threads')
        y_pred = np.empty(X.shape[0], dtype=np.int32)
        for i in range(X.shape[0]):
            y_pred[i] = self._predict(X[i])
        return y_pred
    
    def _predict(self, x):
        # Calculate distances from the input point to all training points
        distances = np.empty(self.X_train.shape[0])
        for i in range(self.X_train.shape[0]):
            distances[i] = self.euclidean_distance(x, self.X_train[i])
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = self.y_train[k_indices]
        # Return the most common class label among the k nearest neighbors
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
