"""
Handle the revision operators.
"""
from abc import ABC

from src.util import Initializable


class RevisionOperator(ABC, Initializable):
    """
    Operator to revise the theory.
    """

    def __init__(self, learning_system, theory_metric):
        self.learning_system = learning_system
        self.theory_metric = theory_metric
