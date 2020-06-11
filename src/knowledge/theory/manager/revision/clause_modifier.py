"""
Modifies proposed clauses.
"""
from abc import abstractmethod

from src.knowledge.examples import Examples
from src.language.language import HornClause
from src.structure_learning.structure_learning_system import \
    StructureLearningSystem
from src.util import Initializable


class ClauseModifier(Initializable):
    """
    A clause modifier.
    """

    def __init__(self, learning_system):
        """
        Creates a clause modifier.

        :param learning_system: the learning system
        :type learning_system: StructureLearningSystem
        """
        self.learning_system = learning_system

    @abstractmethod
    def modify_clause(self, clause):
        """
        Modifies the clause, given the examples.

        :param clause: the clause
        :type clause: HornClause
        :param examples: the examples
        :type examples: Examples
        :return: a new clause
        :rtype: HornClause
        """
        pass
