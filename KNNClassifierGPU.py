import numpy as np
from numba import cuda, float64, int32
import math

class KNNClassifierGPU:
    def __init__(self, k=3):
        self.k = k
        self.X_train = np.empty((0, 0), dtype=np.float64)
        self.y_train = np.empty(0, dtype=np.int32)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, blocks_per_grid, threads_per_block):
        # Allocate space for the predictions
        y_pred = np.empty(X.shape[0], dtype=np.int32)

        # Allocate space for distances on the device
        distances = np.empty((X.shape[0], self.X_train.shape[0]), dtype=np.float64)
        
        # Copy data to device
        d_X_train = cuda.to_device(self.X_train)
        d_y_train = cuda.to_device(self.y_train)
        d_X = cuda.to_device(X)
        d_distances = cuda.to_device(distances)

        # print number of block dimensions as well as threads per block
        print(f'blocks_per_grid : {blocks_per_grid} x {blocks_per_grid} - threads_per_block {threads_per_block}')
        # totale number of threads
        print(f'Number of threads {blocks_per_grid*blocks_per_grid*threads_per_block*threads_per_block}')

        # Run the kernel to calculate distances
        compute_distances_kernel[(blocks_per_grid, blocks_per_grid), (threads_per_block, threads_per_block)](d_X, d_X_train, d_distances)

        # Copy distances back to host
        d_distances.copy_to_host(distances)

        # For each test point, find the k nearest neighbors on the host
        for i in range(X.shape[0]):
            y_pred[i] = self._predict(distances[i])

        return y_pred

    def _predict(self, distances):
        # Sort distances and select the k-nearest labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Determine the most common label in the k-nearest labels
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common


@cuda.jit
def compute_distances_kernel(X, X_train, distances):
    """
    CUDA kernel to compute the Euclidean distance between each test point and each training point.
    Args:
        X: Test data (m x n), where m is the number of test samples and n is the number of features.
        X_train: Training data (N x n), where N is the number of training samples and n is the number of features.
        distances: Output array to store distances (m x N).
    """
    test_idx = cuda.grid(1)  # Current test point based on thread ID
    if test_idx < X.shape[0]:  # Ensure within bounds
        for train_idx in range(X_train.shape[0]):
            dist = 0.0
            for feature_idx in range(X.shape[1]):
                diff = X[test_idx, feature_idx] - X_train[train_idx, feature_idx]
                dist += diff * diff
            distances[test_idx, train_idx] = math.sqrt(dist)