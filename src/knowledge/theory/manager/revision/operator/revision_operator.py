"""
Handle the revision operators.
"""
from abc import abstractmethod

from src.util import Initializable


# TODO: implement this class
class RevisionOperator(Initializable):
    """
    Operator to revise the theory.
    """

    def __init__(self, learning_system, theory_metric):
        self.learning_system = learning_system
        self.theory_metric = theory_metric

    # noinspection PyMissingOrEmptyDocstring
    @abstractmethod
    def required_fields(self):
        pass

    def theory_revision_accepted(self, revised_theory):
        """
        Method to send a feedback to the revision operator, telling
        that the
        revision was accepted.

        :param revised_theory: the revised theory
        :type revised_theory: NeuralLogProgram
        """
        pass
