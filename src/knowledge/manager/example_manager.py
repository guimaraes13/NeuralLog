"""
Manages the incoming examples.
"""
import collections
import logging
from abc import abstractmethod

import src.knowledge.theory.manager.revision.revision_examples as rev_ex
from src.knowledge.examples import Examples
import src.knowledge.theory.manager.revision.sample_selector as selector
from src.language.language import Atom
import src.structure_learning.structure_learning_system as sls
from src.util import Initializable

logger = logging.getLogger(__name__)


class IncomingExampleManager(Initializable):
    """
    Responsible for receiving the examples and suggesting the structure
    learning system to revise the theory, whenever it judges its is necessary.
    """

    OPTIONAL_FIELDS = {"sample_selector": None}

    def __init__(self, learning_system=None, sample_selector=None):
        """
        Creates the a IncomingExampleManager.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param sample_selector: a sample selector
        :type sample_selector: selector.SampleSelector
        """
        self.learning_system = learning_system
        self.sample_selector = sample_selector

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["learning_system"]

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        if self.sample_selector is None:
            self.sample_selector = selector.AllRelevantSampleSelector()
        self.sample_selector.learning_system = self.learning_system
        self.sample_selector.initialize()

    @abstractmethod
    def incoming_examples(self, examples):
        """
        Decide what to do with the incoming `examples`.

        :param examples: the incoming examples
        :type examples: Atom or collections.Iterable[Atom]
        """
        pass

    @abstractmethod
    def get_remaining_examples(self):
        """
        Gets the remaining examples that were not used on the revision.

        :return: the remaining examples
        :rtype: Examples
        """
        pass


class ReviseAllIncomingExample(IncomingExampleManager):
    """
    Class to revise all incoming examples as they arrive.
    """

    # noinspection PyMissingOrEmptyDocstring
    def incoming_examples(self, examples):
        revision_examples = rev_ex.RevisionExamples(self.learning_system,
                                                    self.sample_selector.copy())
        if isinstance(examples, collections.Iterable):
            size = 0
            for example in examples:
                revision_examples.add_example(example)
                size += 1
        else:
            size = 1
            revision_examples.add_example(examples)
        if not size:
            return
        logger.info("Calling revision with %d examples", size)
        self.learning_system.revise_theory(revision_examples)

    # noinspection PyMissingOrEmptyDocstring
    def get_remaining_examples(self):
        return Examples()
