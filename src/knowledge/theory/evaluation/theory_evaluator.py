"""
Evaluates the theory.
"""
import collections
from collections import OrderedDict
from typing import List, Dict, Iterable

import src.structure_learning.structure_learning_system as sls
from src.knowledge.examples import ExamplesInferences, Examples
from src.knowledge.program import NeuralLogProgram
from src.knowledge.theory.evaluation.metric.theory_metric import TheoryMetric
from src.util import Initializable


class TheoryEvaluator(Initializable):
    """
    Responsible for evaluating the theory.
    """

    def __init__(self, learning_system=None, theory_metrics=None):
        """
        Creates a theory evaluator.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param theory_metrics: the theory metrics
        :type theory_metrics: TheoryMetric or collections.Iterable[TheoryMetric]
        """
        self.learning_system: sls.StructureLearningSystem = learning_system
        if theory_metrics is not None:
            if not isinstance(theory_metrics, collections.Iterable):
                self.theory_metrics: List[TheoryMetric] = [theory_metrics]
            else:
                self.theory_metrics: List[TheoryMetric] = list(theory_metrics)

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        for theory_metric in self.theory_metrics:
            theory_metric.initialize()

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["learning_system", "theory_metrics"]

    def evaluate(self, examples, inferred_values=None):
        """
        Evaluates the examples based on the inferred examples, for all the
        theory metrics.

        :param examples: the examples
        :type examples: Examples
        :param inferred_values: the inference of the examples
        :type inferred_values: ExamplesInferences or None
        :return: the evaluation of the theory on the examples, for each
        theory metric
        :rtype: Dict[TheoryMetric, float]
        """
        if inferred_values is None:
            inferred_values = self.learning_system.infer_examples(examples)
        return OrderedDict(
            map(lambda x: (x, x.compute_metric(examples, inferred_values)),
                self.theory_metrics)
        )

    def evaluate_theory(self, examples, metric, theory=None):
        """
        Evaluates the theory on `examples`, using the `metric`. If `theory` is
        not `None`, use it as the theory to be evaluated, instead of the
        current theory of the engine system.

        :param examples: the examples
        :type examples: Examples
        :param metric: the metric
        :type metric: TheoryMetric
        :param theory: If not `None`, use this theory instead of the current
        theory of the engine system
        :type theory: NeuralLogProgram
        :return: the evaluation of the theory
        :rtype: float
        """
        return metric.compute_metric(
            examples,
            self.learning_system.infer_examples(
                examples, theory=theory,
                retrain=metric.parameters_retrain))

    def evaluate_theory_appending_clause(self, examples, metric, clauses):
        """
        Evaluates the theory on `examples`, using the `metric`, appending the
        clauses to the current theory. The appended clauses is used only to
        evaluate the theory, and are then discarded.

        :param examples: the examples
        :type examples: Examples
        :param metric: the metric
        :type metric: TheoryMetric
        :param clauses: the clauses to be appended
        :type clauses: Iterable[Clause]
        :return: the evaluation of the theory
        :rtype: float
        """
        return metric.compute_metric(
            examples, self.learning_system.infer_examples_appending_clauses(
                examples, clauses,
                retrain=metric.parameters_retrain))
