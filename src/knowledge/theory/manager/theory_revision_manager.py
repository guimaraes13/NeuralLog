"""
Manages the revision of the theory.
"""

from abc import ABC, abstractmethod

from src.util import Initializable


class TheoryRevisionManager(Initializable):
    """
    Represents a theory revision manager.
    """

    # noinspection PyMissingOrEmptyDocstring
    @abstractmethod
    def initialize(self) -> None:
        pass

    def __init__(self, learning_system, revision_manager):
        self.learning_system = learning_system
        self.revision_manager = revision_manager
