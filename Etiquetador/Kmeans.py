__authors__ = ['1604517', '1671727', '1673129']
__group__ = '08'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
        Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
        """

        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    def _init_X(self, X):
        """
        Initialization of all pixels, sets X as an array of data in vector
                form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of
                    the sample space is the length of the last dimension
        """

        if X.dtype.kind != 'f':
            X = X.astype(float)

        if len(X.shape) == 3:
            X = np.reshape(X, (X.shape[0] * X.shape[1], 3))

        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """

        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options
        # dictionary
        self.options = options

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        
        """
        seguir mentres l'allargada de la llista de centroides sigui mes petita que la k
        comences amb l'index zero, vas agafant elements de la matriu x i comproves que no estiguin ja posats a la llista
        incrementes l'index perque pugui agafar el segent element de la matriu 
        """

        if self.options['km_init'].lower() == 'first':
            centroids = []
            index = 0
            while len(centroids) < self.K:
                aux = self.X[index].tolist()
                if aux not in centroids:
                    centroids.append(aux)
                index += 1
            self.centroids = np.array(centroids)
            self.old_centroids = np.zeros((self.K, self.X.shape[1]))

        elif self.options['km_init'].lower() == 'random':
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])


        #igual que a firts pero fas un flip de la  matriu x perque agafi els ultims elements. 
        elif self.options['km_init'].lower() == 'last':
            centroids = []
            new_X = np.flip(self.X, axis=0)
            index = 0
            while len(centroids) < self.K:
                aux = new_X[index].tolist()
                if aux not in centroids:
                    centroids.append(aux)
                index += 1

            self.centroids = np.array(centroids)
            self.old_centroids = np.zeros((self.K, self.X.shape[1]))
            
        
        #calcula l'index que 
        elif self.options['km_init'].lower() == 'extreme':
            centroids = []
            #diferencia de cada punt amb el primer punt i es queda amb la mes llunyana
            first_centroid = np.argmax(
                np.sum((self.X - self.X[0]) ** 2, axis=1)) 
            centroids.append(self.X[first_centroid])
            
            second_centroid = np.argmax(
                np.sum((self.X - self.X[first_centroid]) ** 2, axis=1))
            centroids.append(self.X[second_centroid])

            while len(centroids) < self.K:
                distances = np.sum(
                    (self.X - np.array(centroids)[:, np.newaxis]) ** 2, axis=2)
                new_centroid_index = np.argmax(np.min(distances, axis=0))
                centroids.append(self.X[new_centroid_index])

            self.centroids = np.array(centroids)
            self.old_centroids = np.zeros((self.K, self.X.shape[1]))

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each
        point to the closest centroid
        """

        dist = distance(self.X, self.centroids)
        self.labels = np.array(np.argmin(dist, axis=1))

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the
        points assigned to the centroid
        """

        self.old_centroids = np.copy(self.centroids)
        for centroid in range(self.K):
            punts = self.X[np.where(self.labels == centroid)]
            self.centroids[centroid] = np.mean(punts, axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """

        return np.allclose(self.old_centroids, self.centroids)

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of
        iterations is smaller than the maximum number of iterations.
        """

        self._init_centroids()
        while not self.converges() and self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering (distancia dins de la classe) MINIMITZAR
        """

        labels_centroids = self.centroids[self.labels]
        restes = (np.subtract(self.X, labels_centroids)) ** 2
        restes_sumades = np.sum(restes, axis=1)

        N = self.X.shape[0]
        if N != 0:
            self.WCD = sum(restes_sumades) / N

    def interClassDistance(self):
        """
         returns the inter class distance of the current clustering (distancia entre les classes) MAXIMITZAR
        """

        # Utilitzem ICD = 1 / (N * (N - 1)) * sum(dist(C_i, C_j) ^ 2)
        N = self.centroids.shape[0]
        if N <= 1:
            self.ICD = 0.0

        distances = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                distances[i, j] = np.linalg.norm(
                    self.centroids[i] - self.centroids[j]) ** 2

        self.ICD = np.sum(distances) / (N * (N - 1))

    def FisherCoefficient(self):
        """
         returns the fisher coefficient of the current clustering MAXIMITZAR
        """

        self.FC = self.WCD / self.ICD

    def find_bestK(self, max_K, llindar):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """

        def heuristica():
            if self.options['fitting'] == 'WCD':
                self.withinClassDistance()
                return self.WCD
            elif self.options['fitting'] == 'ICD':
                self.interClassDistance()
                return self.ICD
            elif self.options['fitting'] == 'FC':
                self.FisherCoefficient()
                return self.FC

        self.K = 1
        self.fit()
        heuristic = heuristica()

        for k in range(2, max_K + 1):
            self.K = k
            self.fit()
            aux = heuristica()

            dec = aux / heuristic
            heuristic = aux

            if self.options['fitting'] == 'WCD' or \
                    self.options['fitting'] == 'FC':
                if 1 - dec <= llindar:
                    self.K = k - 1
                    return k - 1
            else:
                if dec - 1 > llindar:
                    self.K = k - 1
                    return k - 1

        self.K = max_K


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids
                        points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    dist = np.empty((X.shape[0], C.shape[0]))
    for index, centroid in enumerate(C):
        dist[:, index] = np.sqrt(np.sum(((X - centroid) ** 2), axis=1))

    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label
    following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid
                                    points)
    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    colorProb = utils.get_color_prob(centroids)
    idx = np.argmax(colorProb, axis=1)

    return utils.colors[idx]
