"""
Evaluates the theory.
"""

from src.util import Initializable


class TheoryEvaluator(Initializable):
    """
    Responsible for evaluating the theory.
    """

    def __init__(self, learning_system, theory_metrics):
        self.learning_system = learning_system
        self.theory_metrics = theory_metrics

    def initialize(self) -> None:
        pass
