"""
Handles the theory metrics.
"""
from abc import ABC, abstractmethod

from src.util import Initializable


class TheoryMetric(ABC, Initializable):
    """
    Class to define the theory metric.
    """

    @abstractmethod
    def evaluate(self) -> float:
        """
        Evaluates the theory according to the metric.

        :return: the evaluation of the theory
        :rtype: float
        """
        pass
