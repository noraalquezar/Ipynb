import numpy as np
import logging
from typing import Tuple, Union
from mylogger import mylogger

logger = mylogger(__name__, logging.ERROR)


class Dataset:
    def __init__(self, X: np.ndarray[float], y: np.ndarray[int]) -> None:
        self.X: np.ndarray[float] = X
        self.y: np.ndarray[int] = y
        self.num_samples: int = X.shape[0]
        self.num_features: int = X.shape[1]

    def most_frequent_label(self) -> int:
        # we find the label that it's repeated the most
        unique, counts = np.unique(self.y, return_counts=True)
        max_count_idx = np.argmax(counts)
        return unique[max_count_idx]

    def random_sampling(self, ratio_samples: float) -> "Dataset":
        # it returns us a dataset with a random subset
        num_samples = self.X.shape[0]
        indices = np.random.choice(num_samples,
                                   size=int(ratio_samples*num_samples),
                                   replace=True)
        logger.info('random sampling correctly done')
        return Dataset(self.X[indices], self.y[indices])

    def split(self, idx: int, val: int) -> Tuple["Dataset", "Dataset"]:
        # splits into two parts a dataset
        left_idx = np.where(self.X[:, idx] <= val)[0]
        right_idx = np.where(self.X[:, idx] > val)[0]
        left_dataset = Dataset(self.X[left_idx, :], self.y[left_idx])
        right_dataset = Dataset(self.X[right_idx, :], self.y[right_idx])
        logger.debug('split done')
        return left_dataset, right_dataset

    def frequency(self) -> np.ndarray[int]:
        # this method calculates the frequency of each y of a dataset
        logger.info('frequency returned')
        return np.bincount(self.y)

    def mean_value(self) -> Union[np.ndarray[float], float]:
        # this method calculates the mean value of each y of a dataset
        if len(self.y) == 0:
            return 0.0
        else:
            logger.info('mean value returned')
            return np.mean(self.y)
