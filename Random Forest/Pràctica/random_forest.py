import numpy as np
import logging
import time
import multiprocessing
import collections
import math
from typing import List, Union
from abc import ABC, abstractmethod
from numpy import ndarray
from node import Leaf, Parent
from dataset import Dataset
from visitor import FeatureImportance, PrinterTree
from node import Node
from impurity_measure import ImpurityMeasure
from mylogger import mylogger

logger = mylogger(__name__, logging.ERROR)


class RandomForest(ABC):
    def __init__(self, max_depth: Union[int, None], min_size: int,
                 ratio_samples: float, num_trees: int, num_random_features: int,
                 impurity_measure: Union[ImpurityMeasure, None]) -> None:

        """
        Parameters:

        max_depth: int or None
            The maximum depth of each tree. It helps control the complexity of
            the trees and prevent overfitting
        min_size: int
            The minimum number of samples required to split on interval node
        ratio_samples: float
            The proportion of samples used for training each tree in the random
            forest
        num_trees: int
            The number of trees in the forest
        num_random_features: int
            The square root of the total number of features
        impurity_measure: ImpurtiyMeasure or None
            The function to measure the quality of a split. Supported
            impurity_measure are Gini, Entropy and SumSquareError
        """

        self.max_depth = max_depth
        self.min_size = min_size
        self.ratio_samples = ratio_samples
        self.num_trees = num_trees
        self.num_random_features = num_random_features
        self.impurity_measure = impurity_measure

    def fit(self, X: np.ndarray[float], y: np.ndarray[int]) -> None:
        # a pair (X,y) is a dataset, with its own responsibilities
        dataset = Dataset(X, y)
        self._make_decision_trees_multiprocessing(dataset)
        logger.info('fit correctly done')

    def predict(self, X: np.ndarray[float]) -> ndarray:
        ypred = []
        for samples in X:
            predictions = [tree.predict(samples)
                           for tree in self.decision_trees]
            # majority voting
            ypred.append(self._combine_predictions(predictions))
        logger.info('prediction correctly done')
        return np.array(ypred)

    def _target(self, dataset: Dataset, nproc: int) -> Node:
        print('process {} starts'.format(nproc))
        subset = dataset.random_sampling(self.ratio_samples)
        tree = self._make_node(subset, 1)  # the root of the decision tree
        print('process {} ends'.format(nproc))
        return tree

    def _make_decision_trees_multiprocessing(self, dataset: Dataset) -> None:
        t1 = time.time()
        with multiprocessing.Pool() as pool:
            self.decision_trees = pool.starmap(self._target,
                                               [(dataset, nprocess)
                                                for nprocess
                                                in range(self.num_trees)])
        t2 = time.time()
        print('{} seconds per tree'.format((t2 - t1) / self.num_trees))
        logger.info('decision tress correctly made')

    def _make_node(self, dataset: Dataset, depth: int) -> Node:
        if depth == self.max_depth or dataset.num_samples <= self.min_size or \
                len(np.unique(dataset.X)) == 1:
            # last condition is true if all samples belong to the same class
            node = self._make_leaf(dataset, depth)
        else:
            node = self._make_parent_or_leaf(dataset, depth)
        logger.info('node made')
        return node

    @abstractmethod
    def _make_leaf(self, dataset: Dataset, depth: int) -> Node:
        pass

    def _make_parent_or_leaf(self, dataset: Dataset, depth: int) -> Node:
        # select a random subset of features, to make trees more diverse
        idx_features = np.random.choice(range(dataset.num_features),
                                        self.num_random_features, replace=False)
        best_feature_index, best_threshold, minimum_cost, best_split = \
            self._best_split(idx_features, dataset)
        left_dataset, right_dataset = best_split
        assert left_dataset.num_samples > 0 or right_dataset.num_samples > 0
        if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
            # this is a special case : dataset has samples of at least two
            # classes but the best split is moving all samples to the left or
            # right dataset and none to the other, so we make a leaf instead
            # of a parent
            logger.debug('enter if to make a leaf')
            return self._make_leaf(dataset, depth)
        else:
            node = Parent(best_feature_index, best_threshold)
            node.left_child = self._make_node(left_dataset, depth + 1)
            node.right_child = self._make_node(right_dataset, depth + 1)
            logger.info('parent made')
            return node

    def feature_importance(self) -> np.ndarray[float]:
        feat_imp_visitor = FeatureImportance()
        for tree in self.decision_trees:
            tree.accept_visitor(feat_imp_visitor)
        logger.info('feature importance made')
        return feat_imp_visitor.occurences

    def print_trees(self) -> None:
        for tree in self.decision_trees:
            tree_printer = PrinterTree(self.max_depth)
            tree.accept_visitor(tree_printer)
        logger.info('print trees made')

    def _best_split(self, idx_features: collections.Iterable, dataset: Dataset)\
            -> [int, int, float, tuple[Dataset, Dataset]]:
        best_k, best_v, min_cost, best_split = None, None, np.Inf, None
        for k in idx_features:
            values = np.unique(dataset.X[:, k])
            maxval = np.max(values)
            minval = np.min(values)
            v = np.random.uniform(minval, maxval)
            left_dataset, right_dataset = dataset.split(k, v)
            if self.impurity_measure is not None:
                cost = self._CART_cost(left_dataset, right_dataset)  # J(k,v)
                if cost < min_cost:
                    best_k, best_v, min_cost, best_split = k, v, cost, \
                        [left_dataset, right_dataset]
            else:
                best_k, best_v, min_cost, best_split = k, v, None, \
                    [left_dataset, right_dataset]
        logger.debug('best split made')
        return best_k, best_v, min_cost, best_split

    def _CART_cost(self, left_dataset: Dataset, right_dataset: Dataset) -> \
            float:
        # best pair minimizes this cost function
        total_samples = left_dataset.num_samples + right_dataset.num_samples
        cost = (left_dataset.num_samples
                * self.impurity_measure.compute(left_dataset)
                + right_dataset.num_samples
                * self.impurity_measure.compute(right_dataset)) / total_samples
        logger.debug('cart cost done')
        return cost

    @abstractmethod
    def _combine_predictions(self, predictions: List[float]) -> \
            np.ndarray[float]:
        pass


class RandomForestClassifier(RandomForest):

    """
    The random forest algorithm works by creating a large number of decision
    trees, each trained on a random subset of the training data and using a
    random subset of the input features. These decision trees are trained
    independently and make predictions based on their individual tree
    structures. The final prediction of the random forest classifier is
    determined by aggregating the predictions of all the individual trees,
    through majority voting for classification.
    """

    def _combine_predictions(self, predictions: List[float]) -> \
            np.ndarray[float]:
        """
        Parameters
        ----------
        predictions : array of floats
            includes the predicted values of the classes

        Returns
        -------
        The most frequent value (label) in the array

        This function takes a list of predicted values and returns the most
        frequent value in the array.
        """

        logger.debug('combine_predictions made correctly')
        return np.argmax(np.bincount(predictions))

    def _make_leaf(self, dataset: Dataset, depth: int) -> Node:

        """
        Parameters:

        dataset: Dataset
                This parameter represents the dataset used to create the leaf
                node.
        depth: int
                The depth of each decision tree.
        Return:
            Leaf

        This function creates a leaf node for a decision tree based on the
        most frequent label of the dataset provided.
        """

        logger.info('leaf made')
        return Leaf(dataset.most_frequent_label())


class RandomForestRegressor(RandomForest):

    """
    The random forest algorithm works by creating a large number of decision
    trees, each trained on a random subset of the training data and using a
    random subset of the input features. These decision trees are trained
    independently and make predictions based on their individual tree
    structures. The final prediction of the random forest classifier is
    determined by aggregating the predictions of all the individual trees,
    through averaging for regression.
    """

    def _combine_predictions(self, predictions: List[float]) -> \
            np.ndarray[float]:
        """

        Parameters
        ----------
        predictions : array of floats
            includes the predicted values of the classes

        Returns
        -------
        The mean of the values (labels) in the array

        This function takes a list of predicted values and returns the mean
        of these values.
        """

        logger.debug('combine_predictions made correctly')
        return np.mean(predictions)

    def _make_leaf(self, dataset: Dataset, depth: int) -> Node:

        """
        Parameters:

        dataset: Dataset
                This parameter represents the dataset used to create the leaf
                node.
        depth: int
                The depth of each decision tree.
        Return:
            Leaf

        This function creates a leaf node for a decision tree based on the
        mean value of the dataset provided.
        """
        logger.info('leaf made')
        return Leaf(dataset.mean_value())


class IsolationForest(RandomForest):

    """
    The main idea behind the isolation forest algorithm is that anomalies are
    often easier to isolate and separate from normal instances in a dataset.
    The algorithm creates a set of isolation trees, which are binary trees
    designed to partition the data points. Each tree is built by randomly
    selecting a feature and then randomly selecting a split value within the
    range of that feature. The process is repeated recursively until all data
    points are isolated and form leaf nodes.
    """

    def __init__(self, num_trees: int, ratio_samples: float) -> None:

        """
        Parameters:

        num_trees: int
            The number of trees in the forest
        ratio_samples: float
            The proportion of samples used for training each tree in the random
            forest
        """

        self.max_depth = None
        self.min_size = 1
        self.num_random_features = 1
        self.impurity_measure = None
        self.train_size = None
        self.test_size = None
        super().__init__(self.max_depth, self.min_size, ratio_samples,
                         num_trees, self.num_random_features,
                         self.impurity_measure)

    def fit(self, X: np.ndarray[float]) -> None:

        """
        Parameters:

        X: array of float

        Return:
            None

        This function fits the array X by creating an array of y's with value 0
        and calling the method fit of the inherited class.
        """

        self.max_depth = math.ceil(math.log2(len(X)))
        self.train_size = len(X)
        y = np.zeros(self.train_size)
        super().fit(X, y)

    def predict(self, X: np.ndarray[float]) -> np.ndarray[int]:

        """
        Parameters:

        X: array of float

        Return:
            array of int

        This function predicts the anomalies of the array X by calling the
        method predict of the inherited class.
        """

        self.test_size = len(X)
        return super().predict(X)

    def _combine_predictions(self, predictions: List[int]) -> float:

        """
        Parameters
        ----------
        predictions: array of ints
            the depths in the decision trees for a given X

        Returns
        -------
        float
            the degree of anomaly between 0 and 1 of a sample correlated to the
            mean of depths

        This function takes a list of depths in the decision trees for a given
        array X, and it returns us the degree of anomaly between 0 and 1 based
        on the mean of the depth.
        """

        ehx = np.mean(predictions)  # mean depth
        cn = 2 * (np.log(self.train_size - 1)
                  + 0.57721) - 2 * (self.train_size - 1) / float(self.test_size)
        return 2 ** (-ehx / cn)

    def _make_leaf(self, dataset: Dataset, depth: int) -> Node:

        """
        Parameters:

        dataset: Dataset
                This parameter represents the dataset used to create the leaf
                node.
        depth: int
                The depth of each decision tree.
        Return:
            Leaf

        This function creates a leaf node for a decision tree based on the
        depth of that tree.
        """

        return Leaf(depth)
