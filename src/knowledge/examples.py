"""
Represents training examples.
"""
from collections import OrderedDict, UserDict
from typing import Dict, Any, MutableMapping

from src.language.language import Predicate, Atom


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
