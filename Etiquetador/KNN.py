__authors__ = ['1604517', '1671727', '1673129']
__group__ = '08'

import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as
                    PxD (P points in a D dimensional space)
        """

        if np.all(np.array(train_data, dtype=object) != float):
            self.train_data = train_data.astype(float)

        self.train_data = train_data.reshape(train_data.shape[0], -1)

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each
        point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points
            in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th
                 test point
        """

        self.test_data = test_data.reshape(test_data.shape[0], -1)
        distances = cdist(self.test_data, self.train_data)

        self.neighbors = []

        for dist in distances:
            index = dist.argsort()[:k]
            self.neighbors.append(self.labels[index])

        self.neighbors = np.array(self.neighbors)

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors
            gets the most voted value
                (i.e. the class at which that row belongs)
        """

        classes = []

        for fila in self.neighbors:
            counter = Counter(fila)
            classe = counter.most_common(1)[0][0]
            classes.append(classe)

        return np.array(classes)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points
            in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape
            for each test image
        """

        self.get_k_neighbours(test_data, k)

        return self.get_class()
