"""
Handles the revision operators on the TreeTheory.
"""
from abc import ABC

from src.knowledge.manager.tree_manager import TreeTheory
from src.knowledge.theory.manager.revision.operator.revision_operator import \
    RevisionOperator


class TreeRevisionOperator(RevisionOperator, ABC):
    """
    Super class for revision operator that performs operation in the TreeTheory.
    """

    def __init__(self, learning_system=None, theory_metric=None,
                 clause_modifiers=None, tree_theory=None):
        """
        Creates a tree revision operator.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param theory_metric: the theory metric
        :type theory_metric: TheoryMetric
        :param clause_modifiers: a clause modifier, a list of clause modifiers
        or none
        :type clause_modifiers: ClauseModifier or Collection[ClauseModifier]
        or None
        :param tree_theory: the tree theory
        :type tree_theory: TreeTheory
        """
        super().__init__(learning_system, theory_metric, clause_modifiers)
        self.tree_theory = tree_theory

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields() + ["tree_theory"]
