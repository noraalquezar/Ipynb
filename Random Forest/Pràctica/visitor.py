import logging
from abc import ABC, abstractmethod
from mylogger import mylogger
from node import Parent, Leaf

logger = mylogger(__name__, logging.ERROR)


class Visitor(ABC):
    @abstractmethod
    def visit_parent(self, parent: Parent) -> None:
        # abstract method that it's implemented in the subclasses
        pass

    @abstractmethod
    def visit_leaf(self, leaf: Leaf) -> None:
        # abstract method that it's implemented in the subclasses
        pass


class FeatureImportance(Visitor):
    def __init__(self) -> None:
        self.occurences = {}

    def visit_parent(self, parent: Parent) -> None:
        # it updates the occurences with the frequency of each feature_index
        k = parent.feature_index
        if k in self.occurences.keys():
            self.occurences[k] += 1
        else:
            self.occurences[k] = 1
        parent.left_child.accept_visitor(self)
        parent.right_child.accept_visitor(self)
        logger.info('visit parent')

    def visit_leaf(self, leaf: Leaf) -> None:
        pass


class PrinterTree(Visitor):
    def __init__(self, depth: int) -> None:
        self.depth = depth

    def visit_parent(self, parent: Parent) -> None:
        # it prints the information of the parent given
        print(' ' * self.depth + 'parent, {}, {}'.format(parent.feature_index,
                                                         parent.threshold))
        self.depth += 1
        parent.left_child.accept_visitor(self)
        parent.right_child.accept_visitor(self)
        self.depth -= 1
        logger.info('visit parent')

    def visit_leaf(self, leaf: Leaf) -> None:
        # it prints the information of the leaf that is visited
        print(' ' * self.depth + 'leaf, {}'.format(leaf.label))
        logger.info('visit leaf')
