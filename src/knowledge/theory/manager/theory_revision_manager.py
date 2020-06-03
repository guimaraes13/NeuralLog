"""
Manages the revision of the theory.
"""

from abc import ABC

from src.util import Initializable


class TheoryRevisionManager(ABC, Initializable):
    """
    Represents a theory revision manager.
    """

    def __init__(self, learning_system, revision_manager):
        self.learning_system = learning_system
        self.revision_manager = revision_manager
