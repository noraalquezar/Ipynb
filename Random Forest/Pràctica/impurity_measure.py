import numpy as np
import logging
from abc import ABC, abstractmethod
from dataset import Dataset
from mylogger import mylogger

logger = mylogger(__name__, logging.ERROR)


class ImpurityMeasure(ABC):
    @abstractmethod
    def compute(self, dataset: Dataset) -> float:
        pass


class Gini(ImpurityMeasure):
    def compute(self, dataset: Dataset) -> float:
        # it computes the Gini value of the Dataset
        probs = dataset.mean_value()
        logger.debug('gini calculated')
        return 1.0 - np.sum(probs**2)


class Entropy(ImpurityMeasure):
    def compute(self, dataset: Dataset) -> float:
        # it computes the Entropy value of the Dataset
        class_probabilities = dataset.frequency() / len(dataset.y)
        entropy = 0
        if class_probabilities.all() > 0:
            entropy = -np.sum(class_probabilities * np.log(class_probabilities))
        logger.debug('entropy calculated')
        return entropy


class SumSquareError(ImpurityMeasure):
    def compute(self, dataset: Dataset) -> np.ndarray[float]:
        # it computes the SEE value of the Dataset
        class_probabilities = dataset.mean_value()
        logger.debug('sse calculated')
        return np.sum((dataset.y - class_probabilities)**2)
