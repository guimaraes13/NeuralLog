"""
Handles the selection of the revision operators.
"""
from abc import ABC, abstractmethod

from src.util import Initializable


class RevisionOperatorSelector(Initializable):
    """
    Class responsible for selecting the best suited revision operator.
    """

    def __init__(self, operator_evaluators):
        self.operator_evaluators = operator_evaluators

    # noinspection PyMissingOrEmptyDocstring
    @abstractmethod
    def initialize(self) -> None:
        pass