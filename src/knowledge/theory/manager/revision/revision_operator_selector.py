"""
Handles the selection of the revision operators.
"""
from abc import abstractmethod

from src.knowledge.theory.evaluation.metric.theory_metric import TheoryMetric
from src.knowledge.theory.manager.revision.revision_operator_evaluator import \
    RevisionOperatorEvaluator
from src.language.language import Predicate, Atom
from src.util import Initializable


# TODO: implement this class
class RevisionOperatorSelector(Initializable):
    """
    Class responsible for selecting the best suited revision operator.
    """

    def __init__(self, operator_evaluators):
        self.operator_evaluators = operator_evaluators

    # noinspection PyMissingOrEmptyDocstring
    @abstractmethod
    def required_fields(self):
        pass

    def select_operator(self, examples, theory_metric):
        """
        Selects the best operator to revise the theory, based on the examples
        and the metric.

        :param examples: the examples
        :type examples: dict[Predicate, dict[Any, Atom]]
        :param theory_metric: the theory metric
        :type theory_metric: TheoryMetric
        :return: the best revision operator
        :rtype: RevisionOperatorEvaluator
        """
        pass
