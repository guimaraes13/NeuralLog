"""
Handles the selection of relevant examples for structure learning, from all
available examples.
"""
from abc import abstractmethod
from typing import Set

import src.knowledge.theory.manager.revision.operator.revision_operator as ro
import src.structure_learning.structure_learning_system as sls
from src.language.language import Atom
from src.util import Initializable


class SampleSelector(Initializable):
    """
    Class to select relevant examples, for structure learning, from all
    available examples.
    """

    def __init__(self, learning_system=None):
        """
        Creates a SampleSelector.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        """
        self.learning_system = learning_system

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


class AllRelevantSampleSelector(SampleSelector):
    """
    Class to select all examples as relevant.
    """

    def __init__(self, learning_system=None):
        """
        Creates a AllRelevantSampleSelector.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
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

    OPTIONAL_FIELDS = dict(SampleSelector.OPTIONAL_FIELDS)
    OPTIONAL_FIELDS.update({
        "relevant_depth": DEFAULT_RELEVANT_DEPTH,
        "infer_relevant": False
    })

    def __init__(self, learning_system=None, relevant_depth=None,
                 infer_relevant=None):
        super().__init__(learning_system)
        self.relevant_depth = relevant_depth
        if self.relevant_depth is None:
            self.relevant_depth: int = self.OPTIONAL_FIELDS["relevant_depth"]

        self.infer_relevant = infer_relevant
        if self.infer_relevant is None:
            self.infer_relevant: bool = self.OPTIONAL_FIELDS["infer_relevant"]

        self.previous_relevant: Set[Atom] = set()

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
            terms, self.relevant_depth, self.learning_system,
            safe_stop=False, infer_relevant=self.infer_relevant)
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
