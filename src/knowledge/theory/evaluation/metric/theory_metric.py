"""
Handles the theory metrics.
"""
import math
from abc import abstractmethod

from src.knowledge.examples import Examples, ExamplesInferences
from src.util import Initializable


class TheoryMetric(Initializable):
    """
    Class to define the theory metric.
    """

    def __init__(self, parameters_retrain_before_evaluate=False):
        """
        Creates a theory metric.

        :param parameters_retrain_before_evaluate: if `True`, the parameters
        will be trained before each candidate evaluation on this metric.
        :type parameters_retrain_before_evaluate: bool
        """
        self.default_value = 0.0
        self.parameters_retrain_before_evaluate = \
            parameters_retrain_before_evaluate

    @abstractmethod
    def compute_metric(self, examples, inferred_values):
        """
        Evaluates the theory according to the metric.

        :param examples: the examples
        :type examples: Examples
        :param inferred_values: the inferred examples
        :type inferred_values: ExamplesInferences
        :return: the evaluation of the theory
        :rtype: float
        """
        pass

    def difference(self, candidate, current):
        """
        Calculates a quantitative difference between the candidate and the
        current evaluations.

        If the candidate is better than the current, it should return a
        positive number, representing how much it is better.

        If the candidate is worst than the current, it should return a
        negative number, representing how much it is worst.

        If they are equal, it should return 0.

        :param candidate: the candidate evaluation
        :type candidate: float
        :param current: the current evaluation
        :type current: float
        :return: the quantitative difference between the candidate and the
        current evaluation
        :rtype: float
        """
        return math.copysign(
            candidate - current, self.compare(candidate, current))

    # noinspection PyMethodMayBeStatic
    def compare(self, o1, o2):
        """
        Compares the two objects. This comparator imposes an ordering that
        are INCONSISTENT with equals; in the sense that two different
        theories might have the same evaluation for a given knowledge base. In
        addition, two equal theories (and its parameters) MUST have the same
        evaluation for the same knowledge base.

        By default, as higher the metric, better the theory. Override this
        method, otherwise.

        :param o1: the first object to compare
        :type o1: float
        :param o2: the second object to compare
        :type o2: float
        :return: `0` if `o1` is equal to `o2`; a value less than `0` if `o1` is
        numerically less than `o2`; a value greater than `0` if `o1` is
        numerically greater than `o2`
        :rtype: int
        """
        return o1 - o2

    @abstractmethod
    def get_range(self):
        """
        Gets the range of the metric. The range of a metric is the difference
        between the maximum and the minimum value a metric can assume.

        :return: the range of the metric
        :rtype: float
        """
        pass

    def get_best_possible_improvement(self, current_value):
        """
        Calculates the best possible improvement over the `current_value`.
        The best possible improvement is the comparison between the maximum
        possible value of the metric against the `current_value`.

        :param current_value: the current value
        :type current_value: float
        :return: the maximum possible improvement over the current value
        :rtype: float
        """
        return self.difference(self.get_maximum_value(), current_value)

    @abstractmethod
    def get_maximum_value(self):
        """
        Gets the maximum value of the metric. The maximum value of a metric
        is the best value the metric can assume.

        :return: the maximum value of the metric
        :rtype: float
        """
        pass

    @abstractmethod
    def __eq__(self, other):
        if self == other:
            return True

        if not isinstance(other, TheoryMetric):
            return False

        if self.__class__.__name__ != other.__class__.__name__:
            return False

        if self.parameters_retrain_before_evaluate != \
                other.parameters_retrain_before_evaluate:
            return False

        return self.default_value == other.default_value

    @abstractmethod
    def __hash__(self):
        return hash(
            (self.parameters_retrain_before_evaluate, self.default_value))

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


# TODO: implement this class and the other metrics
class RocCurveMetric(TheoryMetric):
    """
    Computes the area under the ROC curve.
    """

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        pass

    # noinspection PyMissingOrEmptyDocstring
    def compute_metric(self, examples, inferred_values):
        pass

    # noinspection PyMissingOrEmptyDocstring
    def difference(self, candidate, current):
        pass

    # noinspection PyMissingOrEmptyDocstring
    def __eq__(self, other):
        pass

    # noinspection PyMissingOrEmptyDocstring
    def __hash__(self):
        pass
