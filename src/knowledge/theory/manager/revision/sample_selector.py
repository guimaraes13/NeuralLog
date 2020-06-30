"""
Handles the selection of relevant examples for structure learning, from all
available examples.
"""
from abc import abstractmethod
from typing import Set

import src.knowledge.theory.manager.revision.operator.revision_operator as ro
from src.language.language import Atom
from src.structure_learning.structure_learning_system import \
    StructureLearningSystem
from src.util import Initializable, reset_field_error, InitializationException


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
    def required_fields(self):
        return ["learning_system"]

    @abstractmethod
    def is_all_relevant(self):
        """
        Returns `True`, if any example are relevant.

        :return: `True`, if any example are relevant.
        :rtype: bool
        """
        pass

    def is_relevant(self, example):
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
    def is_all_relevant(self):
        return True

    # noinspection PyMissingOrEmptyDocstring
    def is_relevant(self, example):
        return True

    # noinspection PyMissingOrEmptyDocstring
    def copy(self):
        return self

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return []


class IndependentSampleSelector(SampleSelector):
    """
    Selects an independent sample of examples from the target of a revision
    point. Two examples are said to be independent, to given distance,
    if they do not share any common relevant atom in the given distance.
    """

    DEFAULT_RELEVANT_DEPTH = 0

    OPTIONAL_FIELDS = SampleSelector.OPTIONAL_FIELDS
    OPTIONAL_FIELDS.update({
        "relevant_depth": DEFAULT_RELEVANT_DEPTH
    })

    def __init__(self, learning_system=None, relevant_depth=None):
        super().__init__(learning_system)
        self._relevant_depth = relevant_depth
        if self._relevant_depth is None:
            self._relevant_depth: int = self.OPTIONAL_FIELDS["relevant_depth"]
        self.previous_relevant: Set[Atom] = set()

    @property
    def relevant_depth(self):
        """
        The relevant depth.
        :return: relevant depth
        :rtype: int
        """
        return self._relevant_depth

    @relevant_depth.setter
    def relevant_depth(self, value):
        if self.previous_relevant:
            raise InitializationException(
                "It is not allowed to reset {}, from {}, "
                "after using it.".format(
                    "relevant_depth", self.__class__.__name__))
        self._relevant_depth = value

    # noinspection PyMissingOrEmptyDocstring
    def is_all_relevant(self):
        return self.relevant_depth < self.DEFAULT_RELEVANT_DEPTH

    # noinspection PyMissingOrEmptyDocstring
    def is_relevant(self, example):
        if self.is_all_relevant():
            return True
        relevant = False
        terms = set(filter(lambda x: x.is_constant(), example.terms))
        current_relevant = ro.relevant_breadth_first_search(
            terms, self.relevant_depth, self.learning_system, False)
        if self.previous_relevant.isdisjoint(current_relevant):
            relevant = True
        self.previous_relevant.update(current_relevant)

        return relevant

    # noinspection PyMissingOrEmptyDocstring
    def copy(self):
        sample_selector = IndependentSampleSelector(
            self.learning_system, self.relevant_depth)
        sample_selector.initialize()
        return sample_selector
