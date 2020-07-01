"""
Represents training examples.
"""
import collections
from collections import OrderedDict, UserDict
from typing import Dict, Any, MutableMapping, Iterable

from src.language.language import Predicate, Atom


class LimitedIterator:
    """
    A limited iterator that iterates over another iterator until a predefined
    number of items or until the iterator is over.

    It can reset to continue iterating over the remaining items.
    """

    def __init__(self, iterator, iterator_size=-1):
        """
        Creates a limited iterator.

        :param iterator: the iterator
        :type iterator: collections.Iterable or collections.Iterator
        :param iterator_size: the size of the iterator if less than or equal
        to zero, iterates until `iterator` is over
        :type iterator_size: int
        """
        if isinstance(iterator, collections.Iterable):
            iterator = iter(iterator)
        self.iterator = iterator
        self.iterator_size = iterator_size
        self.current_item = 0
        self.has_next = True

    def reset(self):
        """
        Resets the iterator to continue iterating for more `iterator_size`.
        """
        self.current_item = 0

    def __next__(self):
        if self.iterator_size < 1 or self.current_item < self.iterator_size:
            try:
                value = next(self.iterator)
            except StopIteration as e:
                self.has_next = False
                raise e
            self.current_item += 1
            return value
        raise StopIteration

    def __iter__(self):
        return self


class ExampleIterator(Iterable[Atom]):
    """
    Iterates over the examples.

    Transforms an Examples object into a collections.Iterable[Atom].
    """

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

    def __init__(self, *args):
        super().__init__(OrderedDict(*args))

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

    def contains_example(self, example):
        """
        Returns `True` if it contains the inference for the `example`.

        :param example: the example
        :type example: Atom
        :return: `True` if it contains the inference for the `example`;
        otherwise, `False`
        :rtype: bool
        """
        if example.predicate not in self:
            return False
        return example.simple_key() in self[example.predicate]

    def add_inference(self, example, value=None):
        """
        Adds the inference for the examples.

        :param example: the example
        :type example: Atom
        :param value: the value
        :type value: Optional[float]
        """
        if value is None:
            value = example.weight
        self.data.setdefault(
            example.predicate,
            OrderedDict())[example.simple_key()] = value

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
