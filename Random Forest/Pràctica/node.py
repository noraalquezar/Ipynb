import numpy as np
from abc import ABC, abstractmethod
# for this class, when we import the class visitor, at the same time, in the
# class visitor, we are importing this class. So a circular import happens.
# For this reason, we have put the Visitor name between quotation marks.


class Node(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray[float]) -> float:
        pass

    @abstractmethod
    def accept_visitor(self, visitor: "Visitor") -> None:
        pass


class Leaf(Node):
    def __init__(self, label: float) -> None:
        self.label = label

    def predict(self, X: np.ndarray[float]) -> float:
        # This predict return us the label(class) that the Leaf has
        return self.label

    def accept_visitor(self, visitor: "Visitor") -> None:
        # This allows the visitor to apply specific behavior or operations to
        # the leaf object
        visitor.visit_leaf(self)


class Parent(Node):
    def __init__(self, feature_index: int, threshold: float) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = None
        self.right_child = None

    def predict(self, X: np.ndarray[float]) -> float:
        # This predict returns us the left_child or the right_child
        # of the parent.
        if X[self.feature_index] < self.threshold:
            return self.left_child.predict(X)
        else:
            return self.right_child.predict(X)

    def accept_visitor(self, visitor: "Visitor") -> None:
        # This allows the visitor to apply specific behavior or operations to
        # the leaf object
        visitor.visit_parent(self)
