"""
Manages the revision of the theory.
"""

from abc import abstractmethod

from src.knowledge.theory.manager.revision.revision_manager import \
    RevisionManager
from src.structure_learning.structure_learning_system import \
    StructureLearningSystem
from src.util import Initializable


class TheoryRevisionManager(Initializable):
    """
    Represents a theory revision manager. The theory revision manager is
    responsible for decide whether a proposed theory should replace the
    current theory of the system.
    """

    def __init__(self, learning_system, revision_manager):
        """
        Creates a theory revision manager.

        :param learning_system: the learning system
        :type learning_system: StructureLearningSystem
        :param revision_manager: the revision manager
        :type revision_manager: RevisionManager
        """
        self.learning_system = learning_system
        self.revision_manager = revision_manager

    # noinspection PyMissingOrEmptyDocstring
    @abstractmethod
    def required_fields(self):
        pass
