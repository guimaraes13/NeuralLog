"""
Handle the evaluation of the revision operators.
"""
from typing import List, Collection

import \
    neurallog.knowledge.theory.manager.revision.operator.revision_operator as ro
from neurallog.knowledge.examples import Examples
from neurallog.knowledge.program import NeuralLogProgram
from neurallog.knowledge.theory.evaluation.metric.theory_metric import \
    TheoryMetric
from neurallog.knowledge.theory.manager.revision.clause_modifier import \
    ClauseModifier
from neurallog.util import Initializable


class RevisionOperatorEvaluator(Initializable):
    """
    Class responsible for evaluating the revision operator.
    """

    OPTIONAL_FIELDS = {
        "updated_theory": None,
        "is_revised": False
    }

    def __init__(self, revision_operator=None):
        """
        Creates a revision operator evaluator.

        :param revision_operator: the revision operator
        :type revision_operator: ro.RevisionOperator or None
        """
        self.revision_operator = revision_operator
        self.updated_theory = self.OPTIONAL_FIELDS["updated_theory"]
        self.is_revised = self.OPTIONAL_FIELDS["is_revised"]

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        self.revision_operator.initialize()

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["revision_operator"]

    def evaluate_operator(self, examples, theory_metric,
                          minimum_threshold=None):
        """
        Evaluates the operator in the examples, based on the metric.

        :param examples: the examples
        :type examples: Examples
        :param theory_metric: the metric
        :type theory_metric: TheoryMetric
        :param minimum_threshold: a minimum threshold to consider by the
        operator. Implementations of this class could use this threshold in
        order to improve performance by skipping evaluating candidates
        :type minimum_threshold: Optional[float]
        :return: the evaluation of the operator
        :rtype: float
        """
        if not self.is_revised:
            self.updated_theory = \
                self.revision_operator.perform_operation(
                    examples, minimum_threshold)
            self.is_revised = True

        if self.updated_theory is None:
            return theory_metric.default_value

        return self.revision_operator.theory_evaluator.evaluate_theory(
            examples, theory_metric, self.updated_theory)

    def get_revised_theory(self, examples):
        """
        Gets the revised theory. This method is useful because most of the 
        revision operators need to previously apply the change before 
        evaluating the theory. This method allows it to store the revised 
        theory, in order to improve performance.
        
        :param examples: the training examples
        :type examples: Examples
        :return: the revised theory
        :rtype: NeuralLogProgram
        """
        if self.is_revised:
            self.is_revised = False
            return self.updated_theory

        return self.revision_operator.perform_operation(examples)

    def theory_revision_accepted(self, revised_theory, examples):
        """
        Method to send a feedback to the revision operator, telling that the
        revision was accepted.

        :param revised_theory: the revised theory
        :type revised_theory: NeuralLogProgram
        :param examples: the examples used on the revision
        :type examples: Examples
        """
        self.revision_operator.theory_revision_accepted(
            revised_theory, examples)

    def clear_cached_theory(self):
        """
        Clears the revised theory.
        """
        self.is_revised = False

    def is_trained(self):
        """
        Checks if the revision operator calls the train of the parameters.

        :return: `True`, if it does; otherwise, `False`
        :rtype: bool
        """
        metric = self.revision_operator.theory_metric
        return metric is not None and metric.parameters_retrain

    @property
    def clause_modifiers(self):
        """
        Gets the clause modifiers.

        :return: the clause modifiers
        :rtype: ClauseModifier or Collection[ClauseModifier] or None
        """
        return self.revision_operator.clause_modifiers

    @clause_modifiers.setter
    def clause_modifiers(self, value):
        """
        Sets the clause modifiers.

        :param value: the clause modifiers
        :type value: ClauseModifier or List[ClauseModifier] or None
        """
        self.revision_operator.clause_modifier = value

    # noinspection PyMissingOrEmptyDocstring
    @property
    def learning_system(self):
        return self.revision_operator.learning_system

    @learning_system.setter
    def learning_system(self, value):
        self.revision_operator.learning_system = value

    def __repr__(self):
        return f"[{self.__class__.__name__}] {self.revision_operator}"
