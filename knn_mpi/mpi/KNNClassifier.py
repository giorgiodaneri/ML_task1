import numpy as np
from mpi4py import MPI


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

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
    
    


class ParallelKNNClassifier:
    
    def __init__(self, k=3):
        self.k = k
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.size = MPI.COMM_WORLD.Get_size()
    
    def fit(self, X, y):
        # Distribute the training data between the MPI processes
        self.all_X = X
        rows = None
        cols = None
        if self.rank == 0:
            rows = X.shape[0]
            cols = X.shape[1]

        self.rows = MPI.COMM_WORLD.bcast(rows, root=0)
        self.cols = MPI.COMM_WORLD.bcast(cols, root=0)
        
        # Compute the distribution of the elements between the MPI processes
        self.rows_per_process = (self.rows // self.size) * np.ones(shape=self.size, dtype=int)
        self.rows_per_process[: self.rows % self.size] += 1

        # Compute the offsets in the distribution of elements
        self.offsets = np.zeros(shape=self.size, dtype=int)
        self.offsets[1:] = np.cumsum(self.rows_per_process[:-1])
        self.offsets[0] = 0

        # Perform a gatherv operation to distribute all training data
        self.X_train = np.empty(shape=(self.rows_per_process[self.rank], self.cols), dtype=float)


        MPI.COMM_WORLD.Scatterv([X, list(self.rows_per_process*self.cols), list(self.offsets*self.cols), MPI.DOUBLE], self.X_train, root=0)
        self.y_train = MPI.COMM_WORLD.bcast(y, root=0)

    def euclidean_distance(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def predict(self, X):
        
        self.X_test = MPI.COMM_WORLD.bcast(X, 0)
        
        y_pred = [self._predict(x) for x in self.X_test]
        return y_pred
    
    def _predict(self, x):

        glob_indices = np.empty(shape=(self.k * 2 * self.size, 1), dtype=float) if self.rank == 0 else None
        loc_distances = np.array([np.linalg.norm(x-x_train) for x_train in self.X_train], dtype=float)
        
        # Now gather all the distances using the offsets
        k_indices = np.argsort(loc_distances)[:self.k]
        k_distances = np.array([[index+self.offsets[self.rank], loc_distances[index]] for index in k_indices]).flatten()
        MPI.COMM_WORLD.Gather(k_distances, glob_indices, root=0)

        if self.rank == 0:
            glob_indices = glob_indices.flatten()
            # Even indices to store the distances, odd indices to store the indices
            indices = glob_indices[0::2]
            distances = glob_indices[1::2]
            to_sort = np.column_stack((distances, indices))
            
            # Sort the to_sort object based on the first column
            sorted_indices = to_sort[to_sort[:, 0].argsort()][:self.k]


            k_nearest_labels = self.y_train[sorted_indices[:, 1].astype(int)]
            most_common = np.bincount(k_nearest_labels).argmax()
            return most_common
        else:
            return None
        