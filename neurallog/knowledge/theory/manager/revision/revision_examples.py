#  Copyright 2021 Victor Guimarães
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Handles the examples to be used on the theory revision.
"""
import sys

import neurallog.structure_learning.structure_learning_system as sls
from neurallog.knowledge.examples import Examples, ExamplesInferences, \
    ExampleIterator
from neurallog.knowledge.theory.manager.revision.sample_selector import \
    SampleSelector
from neurallog.language.language import Atom
from neurallog.util import time_measure


class RevisionExamples:
    """
    Contains the examples to be used on the theory revision.
    """

    def __init__(self, learning_system, sample_selector):
        """
        Creates the revision examples.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param sample_selector: the sample selector
        :type sample_selector: SampleSelector
        """
        self.learning_system = learning_system
        self.sample_selector = sample_selector

        self.incoming_examples = Examples()
        self.relevant_examples = Examples()
        self.inferred_values = ExamplesInferences()
        self.not_evaluated_examples = Examples()
        self.last_inference = -sys.float_info.max

    def get_training_examples(self, all_examples=True):
        """
        Gets the training examples. If `all_examples` is `True`, returns all
        the stored examples; otherwise `False`, returns only the relevant
        examples, based on the sample selector.

        :param all_examples: If it is to return all stored examples or only
        the relevant ones
        :type all_examples: bool
        :return: the examples
        :rtype: Examples
        """
        if all_examples:
            return self.incoming_examples
        else:
            return self.relevant_examples

    def get_number_of_examples(self, all_examples=True, predicate=None):
        """
        Gets the number of examples. If `all_examples` is `True`, return
        the number of all examples; otherwise, returns only the number of
        relevant examples. If `predicate` is not `None`, computes only the
        examples of the given predicate.

        :param all_examples: if `True`, return the number of all examples;
        otherwise, returns only the number of relevant examples.
        :type all_examples: bool
        :param predicate: the predicate of the examples
        :type predicate: Predicate or None
        :return: the number of relevant examples
        :rtype: int
        """
        if all_examples:
            return self.incoming_examples.size(predicate)
        else:
            return self.relevant_examples.size(predicate)

    def add_example(self, example, inferred_value=None):
        """
        Adds the examples to the set of revision examples.

        :param example: the example
        :type example: Atom
        :param inferred_value: the inferred value, if known
        :type inferred_value: Optional[float]
        """
        self.incoming_examples.add_example(example)
        if self.sample_selector.is_relevant(example):
            self.relevant_examples.add_example(example)
            if inferred_value is None:
                self.not_evaluated_examples.add_example(example)
            else:
                self.inferred_values.add_inference(example, inferred_value)

    def add_examples(self, examples):
        """
        Adds the examples to the set of revision examples.

        :param examples: the examples
        :type examples: collections.Iterable[Atom] or Examples
        :return: `True`, if at least one examples has been added; otherwise,
        `False`
        :rtype: bool
        """
        if isinstance(examples, Examples):
            examples = ExampleIterator(examples)

        added = False
        for example in examples:
            self.add_example(example)
            added = True

        return added

    def get_inferred_values(self, last_theory_change):
        """
        Gets the inferred examples. If `last_theory_change` is greater than
        the last inference of the examples, it is re-inferred.

        :param last_theory_change: the last change of the theory
        :type last_theory_change: float
        :return: the inferred examples
        :rtype: ExamplesInferences
        """
        if self.last_inference < last_theory_change:
            self.clear_inferred_values()
        if self.not_evaluated_examples:
            self.inferred_values.update(
                self.learning_system.infer_examples(
                    self.not_evaluated_examples))
            self.not_evaluated_examples.clear()
            self.last_inference = time_measure.performance_time()

        return self.inferred_values

    def clear_inferred_values(self):
        """
        Clears the values of the inferred examples.
        """
        self.inferred_values.clear()
        self.not_evaluated_examples.update(self.relevant_examples)

    def is_empty(self):
        """
        Checks if the set of revision examples is empty.

        :return: `True`, if the set is empty; otherwise, `False`
        :rtype: bool
        """
        return not self.incoming_examples

    def __bool__(self):
        return bool(self.incoming_examples)

    def contains(self, example):
        """
        Checks if it contains the example.

        :param example: the example
        :type example: Atom
        :return: `True` if it contains the example; otherwise, `False`
        :rtype: bool
        """
        return self.incoming_examples.contains(example)
