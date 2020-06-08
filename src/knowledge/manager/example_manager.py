"""
Manages the incoming examples.
"""
import logging
from abc import abstractmethod

from src.knowledge.theory.manager.revision.sample_selector import SampleSelector
from src.structure_learning.structure_learning_system import \
    StructureLearningSystem
from src.util import Initializable, unset_fields_error, reset_field_error

logger = logging.getLogger(__name__)


class IncomingExampleManager(Initializable):
    """
    Responsible for receiving the examples and suggesting the structure
    learning system to revise the theory, whenever it judges its is necessary.
    """

    def __init__(self, learning_system=None, sample_selector=None):
        """
        Creates the a IncomingExampleManager.

        :param learning_system: the learning system
        :type learning_system: StructureLearningSystem
        :param sample_selector: a sample selector
        :type sample_selector: SampleSelector
        """
        self._learning_system = learning_system
        self._sample_selector = sample_selector

    # noinspection PyMissingOrEmptyDocstring
    @abstractmethod
    def initialize(self) -> None:
        logger.debug("Initializing IncomingExampleManager:\t%s",
                     self.__class__.__name__)
        fields = []
        if self.learning_system is None:
            fields.append("learning_system")

        if self.sample_selector is None:
            fields.append("sample_selector")

        if len(fields) > 0:
            raise unset_fields_error(fields, self)

        self.sample_selector.learning_system = self.learning_system
        self.sample_selector.initialize()

    @abstractmethod
    def incoming_examples(self, examples):
        """
        Decide what to do with the incoming `examples`.

        :param examples: the incoming examples
        :type examples: Atom or collection.Iterable[Atom]
        """
        pass

    # noinspection PyMissingOrEmptyDocstring
    @property
    def learning_system(self):
        return self._learning_system

    @learning_system.setter
    def learning_system(self, value):
        if self._learning_system is not None:
            raise reset_field_error("learning_system", self)
        self._learning_system = value

    # noinspection PyMissingOrEmptyDocstring
    @property
    def sample_selector(self):
        return self._sample_selector

    @sample_selector.setter
    def sample_selector(self, value):
        if self._sample_selector is not None:
            raise reset_field_error("sample_selector", self)
        self._sample_selector = value


class ReviseAllIncomingExample(IncomingExampleManager):
    """
    Class to revise all incoming examples as they arrive.
    """

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self) -> None:
        pass

    # noinspection PyMissingOrEmptyDocstring
    def incoming_examples(self, examples):
        # TODO: create the RevisionExamples class
        self.learning_system.revise_theory(examples)
