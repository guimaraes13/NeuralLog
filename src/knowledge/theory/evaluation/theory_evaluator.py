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

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self) -> None:
        super().initialize()

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["learning_system", "theory_metrics"]
