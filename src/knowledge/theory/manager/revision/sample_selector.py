"""
Handles the selection of relevant examples for structure learning, from all
available examples.
"""
from abc import abstractmethod

from src.structure_learning.structure_learning_system import \
    StructureLearningSystem
from src.util import Initializable, unset_fields_error, reset_field_error


class SampleSelector(Initializable):
    """
    Class to select relevant examples, for structure learning, from all
    available examples.
    """

    def __init__(self, learning_system=None):
        """
        Creates a SampleSelector.

        :param learning_system: the learning system
        :type learning_system: StructureLearningSystem
        """
        self._learning_system = learning_system

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self) -> None:
        if self.learning_system is None:
            raise unset_fields_error("learning_system", self)

    @abstractmethod
    def is_all_relevant(self) -> bool:
        """
        Returns `True`, if any example are relevant.

        :return: `True`, if any example are relevant.
        :rtype: bool
        """
        pass

    def is_relevant(self, example) -> bool:
        """
        Returns `True`, if the `example` is relevant.

        :param example: the example
        :type example: Atom
        :return: `True`, if the `example` is relevant.
        :rtype: bool
        """
        pass

    @abstractmethod
    def copy(self):
        """
        Returns a copy of the object.

        :return: the copy of the object
        :rtype: SampleSelector
        """
        pass

    # noinspection PyMissingOrEmptyDocstring
    @property
    def learning_system(self):
        return self._learning_system

    @learning_system.setter
    def learning_system(self, value):
        if self._learning_system is not None:
            raise reset_field_error(self, "learning_system")
        self._learning_system = value


class AllRelevantSampleSelect(SampleSelector):
    """
    Class to select all examples as relevant.
    """

    def __init__(self, learning_system=None):
        """
        Creates a AllRelevantSampleSelector.

        :param learning_system: the learning system
        :type learning_system: StructureLearningSystem
        """
        super().__init__(learning_system)

    # noinspection PyMissingOrEmptyDocstring
    def is_all_relevant(self) -> bool:
        return True

    # noinspection PyMissingOrEmptyDocstring
    def is_relevant(self, example) -> bool:
        return True

    # noinspection PyMissingOrEmptyDocstring
    def copy(self):
        return self

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return None
