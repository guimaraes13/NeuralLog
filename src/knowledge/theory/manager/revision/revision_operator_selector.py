"""
Handles the selection of the revision operators.
"""
import logging
from abc import abstractmethod, ABC
from collections import Collection

from src.knowledge.examples import Examples
from src.knowledge.theory import TheoryRevisionException
from src.knowledge.theory.evaluation.metric.theory_metric import TheoryMetric
from src.knowledge.theory.manager.revision.revision_operator_evaluator import \
    RevisionOperatorEvaluator
from src.util import Initializable

logger = logging.getLogger(__name__)


class RevisionOperatorSelector(Initializable):
    """
    Class responsible for selecting the best suited revision operator.
    """

    def __init__(self, operator_evaluators=None):
        """
        Creates a revision operator selector.

        :param operator_evaluators: the operator evaluators
        :type operator_evaluators: Collection[RevisionOperatorEvaluator] or None
        """
        self.operator_evaluators = operator_evaluators

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["operator_evaluators"]

    @abstractmethod
    def select_operator(self, examples, theory_metric):
        """
        Selects the best operator to revise the theory, based on the examples
        and the metric.

        :param examples: the examples
        :type examples: Examples
        :param theory_metric: the theory metric
        :type theory_metric: TheoryMetric
        :return: the best revision operator
        :rtype: RevisionOperatorEvaluator
        """
        pass


class SelectFirstRevisionOperator(RevisionOperatorSelector):
    """
    Selects the first operator evaluator to revise the theory.
    """

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def initialize(self):
        super().initialize()
        self._operator = next(iter(self.operator_evaluators))

    # noinspection PyMissingOrEmptyDocstring
    def select_operator(self, examples, theory_metric):
        return self._operator


class BestRevisionOperatorSelector(RevisionOperatorSelector):
    """
    Selects the best possible revision operator.
    """

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def initialize(self):
        super().initialize()
        if len(self.operator_evaluators) < 2:
            self.selector: RevisionOperatorEvaluatorSelector = \
                SingleRevisionOperatorEvaluator(self.operator_evaluators)
        else:
            self.selector: RevisionOperatorEvaluatorSelector = \
                BestSelector(self.operator_evaluators)

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def select_operator(self, examples, theory_metric):
        return self.selector.select_operator(examples, theory_metric)


class RevisionOperatorEvaluatorSelector(ABC):
    """
    Class to select the proper operator, given the target examples and the
    metric.
    """

    @abstractmethod
    def select_operator(self, targets, metric):
        """
        Selects the proper operator, based on the target examples and the
        metric.

        :param targets: the target examples
        :type targets: Examples
        :param metric: the metric
        :type metric: TheoryMetric
        :return: the proper revision operator evaluator
        :rtype: RevisionOperatorEvaluator
        """
        pass


class SingleRevisionOperatorEvaluator(RevisionOperatorEvaluatorSelector):
    """
    Selects the only operator.
    """

    def __init__(self, operator_evaluators):
        """
        Create a single revision operator selector.

        :param operator_evaluators: the operator evaluators
        :type operator_evaluators: Collection[RevisionOperatorEvaluator]
        """
        self.operator_evaluator = next(iter(operator_evaluators))

    # noinspection PyMissingOrEmptyDocstring
    def select_operator(self, targets, metric):
        return self.operator_evaluator


class BestSelector(RevisionOperatorEvaluatorSelector):
    """
    Selects the best possible operator.
    """

    def __init__(self, operator_evaluators):
        """
        Create a best operator selector.

        :param operator_evaluators: the operator evaluators
        :type operator_evaluators: Collection[RevisionOperatorEvaluator]
        """
        self.operator_evaluators = operator_evaluators
        self.preferred_operator = next(iter(operator_evaluators))

    # noinspection PyMissingOrEmptyDocstring
    def select_operator(self, targets, metric):
        best_evaluator = self.preferred_operator
        best_evaluation = metric.default_value

        for evaluator in self.operator_evaluators:
            try:
                evaluator.clear_cached_theory()
                current = evaluator.evaluate_operator(targets, metric)
                if metric.compare(current, best_evaluation) > 0:
                    best_evaluation = current
                    best_evaluator = evaluator
            except TheoryRevisionException:
                logger.exception(
                    "Error when evaluating the revision operator, reason:")

        return best_evaluator
