"""
Handles the theory metrics.
"""
from abc import abstractmethod

from src.knowledge.examples import Examples, ExamplesInferences
from src.util import Initializable


class TheoryMetric(Initializable):
    """
    Class to define the theory metric.
    """

    @abstractmethod
    def evaluate(self, examples, inferred_examples):
        """
        Evaluates the theory according to the metric.

        :param examples: the examples
        :type examples: Examples
        :param inferred_examples: the inferred examples
        :type inferred_examples: ExamplesInferences
        :return: the evaluation of the theory
        :rtype: float
        """
        pass

    @abstractmethod
    def difference(self, candidate_evaluation, current_evaluation):
        """
        Calculates a quantitative difference between the candidate and the
        current evaluations.

        If the candidate is better than the current, it should return a
        positive number, representing how much it is better.

        If the candidate is worst than the current, it should return a
        negative number, representing how much it is worst.

        If they are equal, it should return 0.

        :param candidate_evaluation: the candidate evaluation
        :type candidate_evaluation: float
        :param current_evaluation: the current evaluation
        :type current_evaluation: float
        :return: the quantitative difference between the candidate and the
        current evaluation
        :rtype: float
        """
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
    def evaluate(self, examples, inferred_examples):
        pass

    # noinspection PyMissingOrEmptyDocstring
    def difference(self, candidate_evaluation, current_evaluation):
        pass
