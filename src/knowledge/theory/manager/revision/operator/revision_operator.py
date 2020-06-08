"""
Handle the revision operators.
"""
from abc import abstractmethod

from src.util import Initializable


class RevisionOperator(Initializable):
    """
    Operator to revise the theory.
    """

    def __init__(self, learning_system, theory_metric):
        self.learning_system = learning_system
        self.theory_metric = theory_metric

    # noinspection PyMissingOrEmptyDocstring
    @abstractmethod
    def initialize(self) -> None:
        pass
