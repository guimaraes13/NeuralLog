"""
Handle the revision operators.
"""
from abc import abstractmethod
from typing import List

from src.knowledge.examples import Examples
from src.knowledge.program import NeuralLogProgram
from src.knowledge.theory.evaluation.metric.theory_metric import TheoryMetric
from src.knowledge.theory.evaluation.theory_evaluator import TheoryEvaluator
from src.knowledge.theory.manager.revision.clause_modifier import ClauseModifier
from src.structure_learning.structure_learning_system import \
    StructureLearningSystem
from src.util import Initializable


# TODO: extend this class
class RevisionOperator(Initializable):
    """
    Operator to revise the theory.
    """

    def __init__(self, learning_system=None, theory_metric=None,
                 clause_modifier=None):
        """
        Creates a revision operator.

        :param learning_system: the learning system
        :type learning_system: StructureLearningSystem
        :param theory_metric: the theory metric
        :type theory_metric: TheoryMetric
        :param clause_modifier: a clause modifier, a list of clause modifiers
        or none
        :type clause_modifier: ClauseModifier or List[ClauseModifier] or None
        """
        self.learning_system = learning_system
        self.theory_metric = theory_metric
        self.clause_modifier = clause_modifier

    # noinspection PyMissingOrEmptyDocstring
    @property
    def theory_evaluator(self) -> TheoryEvaluator:
        return self.learning_system.theory_evaluator

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["learning_system", "theory_metric"]

    @abstractmethod
    def perform_operation(self, targets):
        """
        Applies the operation on the theory, given the target examples.

        :param targets: the target examples
        :type targets: Examples
        :return: the revised theory
        :rtype: NeuralLogProgram or None
        """
        pass

    @abstractmethod
    def theory_revision_accepted(self, revised_theory):
        """
        Method to send a feedback to the revision operator, telling
        that the
        revision was accepted.

        :param revised_theory: the revised theory
        :type revised_theory: NeuralLogProgram
        """
        pass
