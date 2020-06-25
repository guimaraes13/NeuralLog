"""
Modifies proposed clauses.
"""
from abc import abstractmethod

import src.structure_learning.structure_learning_system as sls
from src.knowledge.examples import Examples
from src.language.language import HornClause
from src.util import Initializable


class ClauseModifier(Initializable):
    """
    A clause modifier.
    """

    def __init__(self, learning_system):
        """
        Creates a clause modifier.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        """
        self.learning_system = learning_system

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["learning_system"]

    @abstractmethod
    def modify_clause(self, clause, examples):
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
