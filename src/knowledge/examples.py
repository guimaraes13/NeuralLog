"""
Represents training examples.
"""
from collections import OrderedDict, UserDict
from typing import Dict, Any, MutableMapping

from src.language.language import Predicate, Atom


class ExampleIterator:
    """Iterates over the examples."""

    def __init__(self, examples, predicate=None):
        """
        Creates an example iterator.

        :param examples: the examples
        :type examples: Examples
        :param predicate: the predicate to iterate over, if `None`, it iterates
        over all examples
        :type predicate: Predicate or None
        """
        self._examples = examples
        if predicate is not None:
            self._outer_iterator = iter([examples.get(predicate, dict())])
        else:
            self._outer_iterator = iter(examples.values())
        self._inner_iterator = None

    def __next__(self) -> Atom:
        if self._inner_iterator is None:
            self._inner_iterator = iter(next(self._outer_iterator).values())
        try:
            return next(self._inner_iterator)
        except StopIteration:
            self._inner_iterator = None
            return self.__next__()

    def __iter__(self):
        return self


class Examples(UserDict, MutableMapping[Predicate, Dict[Any, Atom]]):
    """
    Handles a set of training examples.
    """

    def __init__(self):
        super().__init__(OrderedDict())

    def add_example(self, example):
        """
        Adds the example.

        :param example: the example
        :type example: Atom
        """
        self.data.setdefault(
            example.predicate, OrderedDict())[example.simple_key()] = example

    def size(self, predicate=None):
        """
        Gets the number of examples for `predicate`. If `predicate` is
        `None`, return the total number of examples.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the number of examples for `predicate`. If `predicate` is
        `None`, return the total number of examples
        :rtype: int
        """
        if predicate is not None:
            return len(self.data.get(predicate, {}))
        else:
            return sum(map(lambda x: len(x), self.data.values()))


class ExamplesInferences(UserDict, MutableMapping[Predicate, Dict[Any, float]]):
    """
    Handles a set of inferences.
    """

    def __init__(self):
        super().__init__(OrderedDict())

    def add_inference(self, example, value):
        """
        Adds the inference for the examples.

        :param example: the example
        :type example: Atom
        :param value: the value
        :type value: float
        """
        self.data.setdefault(
            example.predicate, OrderedDict())[example.simple_key()] = value

    def get_value_for_example(self, example):
        """
        Returns the inference value for the examples, if exists.

        :param example: the example
        :type example: Atom
        :return: the inference value, if exists; otherwise, returns `None`
        :rtype: float or None
        """
        return \
            self.get(example.predicate, dict()).get(example.simple_key(), 0.0)
