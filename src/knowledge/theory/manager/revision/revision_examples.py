"""
Handles the examples to be used on the theory revision.
"""
import collections
from collections import OrderedDict
from typing import Dict, Any

from src.knowledge.theory.manager.revision.sample_selector import SampleSelector
from src.language.language import Atom, Predicate
from src.structure_learning.structure_learning_system import \
    StructureLearningSystem
from src.util import time_measure


class RevisionExamples:
    """
    Contains the examples to be used on the theory revision.
    """

    def __init__(self, learning_system, sample_selector):
        """
        Creates the revision examples.

        :param learning_system: the learning system
        :type learning_system: StructureLearningSystem
        :param sample_selector: the sample selector
        :type sample_selector: SampleSelector
        """
        self.learning_system = learning_system
        self.sample_selector = sample_selector

        self.incoming_examples: Dict[Predicate, Dict[Any, Atom]] = OrderedDict()
        self.relevant_examples: Dict[Predicate, Dict[Any, Atom]] = OrderedDict()
        self.inferred_examples: Dict[Predicate, Dict[Any, float]] = \
            OrderedDict()
        self.not_evaluated_examples: Dict[Predicate, Dict[Any, Atom]] = \
            OrderedDict()
        self.last_inference = 0.0

    def get_training_examples(self, all_examples=True):
        """
        Gets the training examples. If `all_examples` is `True`, returns all
        the stored examples; otherwise `False`, returns only the relevant
        examples, based on the sample selector.

        :param all_examples: If it is to return all stored examples or only
        the relevant ones
        :type all_examples: bool
        :return: the examples
        :rtype: dict[Predicate, dict[Any, Atom]]
        """
        return \
            self.incoming_examples if all_examples else self.relevant_examples

    def get_relevant_sample_size(self, predicate=None):
        """
        Gets the number of relevant examples, based on the sample selector.
        If `predicate` is not `None`, computes only the examples of the given
        predicate.

        :param predicate: the predicate of the examples
        :type predicate: Predicate or None
        :return: the number of relevant examples
        :rtype: int
        """
        pass

    def add_example(self, example, inferred_value=None):
        """
        Adds the examples to the set of revision examples.

        :param example: the example
        :type example: Atom
        :param inferred_value: the inferred value of the example
        :type inferred_value: float or None
        """
        predicate = example.predicate
        key = example.simple_key()
        self.incoming_examples.setdefault(
            predicate, OrderedDict())[key] = example
        if self.sample_selector.is_relevant(example):
            self.relevant_examples.setdefault(
                predicate, OrderedDict())[key] = example
            if inferred_value is None:
                self.not_evaluated_examples.setdefault(
                    predicate, OrderedDict())[key] = example
            else:
                self.inferred_examples.setdefault(
                    predicate, OrderedDict())[key] = inferred_value

    def add_examples(self, examples):
        """
        Adds the examples to the set of revision examples.

        :param examples: the examples
        :type examples: collections.Iterable[Atom]
        """
        for example in examples:
            self.add_example(example)

    def get_inferred_examples(self, last_theory_change):
        """
        Gets the inferred examples. If `last_theory_change` is greater than
        the last inference of the examples, it is re-inferred.

        :param last_theory_change: the last change of the theory
        :type last_theory_change: float
        :return: the inferred examples
        :rtype: Dict[Predicate, Dict[Any, float]]
        """
        if self.last_inference < last_theory_change:
            self.clear_inferred_examples()
        if self.not_evaluated_examples:
            self.inferred_examples.update(
                self.learning_system.infer_examples(
                    self.not_evaluated_examples))
            self.not_evaluated_examples.clear()
            self.last_inference = time_measure.performance_time()

        return self.inferred_examples

    def clear_inferred_examples(self):
        """
        Clears the values of the inferred examples.
        """
        self.relevant_examples.clear()
        self.not_evaluated_examples.update(self.relevant_examples)

    def is_empty(self):
        """
        Checks if the set of revision examples is empty.

        :return: `True`, if the set is empty; otherwise, `False`
        :rtype: bool
        """
        return not self.incoming_examples
