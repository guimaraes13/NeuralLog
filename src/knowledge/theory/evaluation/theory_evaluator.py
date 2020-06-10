"""
Evaluates the theory.
"""
import collections

from src.knowledge.theory.evaluation.metric.theory_metric import TheoryMetric
from src.structure_learning.structure_learning_system import \
    StructureLearningSystem
from src.util import Initializable


class TheoryEvaluator(Initializable):
    """
    Responsible for evaluating the theory.
    """

    def __init__(self, learning_system=None, theory_metrics=None):
        """
        Creates a theory evaluator.

        :param learning_system: the learning system
        :type learning_system: StructureLearningSystem
        :param theory_metrics: the theory metrics
        :type theory_metrics: TheoryMetric or collections.Iterable[TheoryMetric]
        """
        self.learning_system = learning_system
        self.theory_metrics: collections.Iterable[TheoryMetric] or None = None
        if theory_metrics is not None:
            if not isinstance(theory_metrics, collections.Iterable):
                theory_metrics = [theory_metrics]
            self.theory_metrics = list(theory_metrics)

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        for theory_metric in self.theory_metrics:
            theory_metric.initialize()

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["learning_system", "theory_metrics"]
