"""
Generates logic variable names.
"""
import re
from collections import deque

GENERATOR_PATTERN = re.compile(r"[A-Z]+")


class VariableGenerator:
    """
    Generate unique logic variable names.
    """

    def __init__(self, avoid_terms=()):
        """
        Creates a logic variable generator.

        :param avoid_terms: a collection of terms to be avoided by the
        variable generator. The generator will not generate terms in the
        avoided collection.
        :type avoid_terms: collection.Collection[str]
        """
        self._possible_values = list(map(lambda x: chr(65 + x), range(26)))
        self._max_index = len(self._possible_values) - 1
        self._pointers = deque([0])
        self._all_avoid_terms = set(
            filter(lambda x: GENERATOR_PATTERN.fullmatch(x) is not None,
                   avoid_terms)
        )
        self._remaining_avoid_terms = set(self._all_avoid_terms)

    # noinspection PyMethodMayBeStatic
    def clean_copy(self):
        """
        Returns a clean copy of the variable generator class.

        :return: A clean copy of this class
        :rtype: VariableGenerator
        """
        return VariableGenerator(self._all_avoid_terms)

    def _increment_pointers(self):
        next_step = 1
        for i in reversed(range(len(self._pointers))):
            if self._pointers[i] == self._max_index:
                self._pointers[i] = 0
                next_step = 1
            else:
                self._pointers[i] += next_step
                next_step = 0
                break
        if next_step > 0:
            self._pointers.appendleft(0)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            string = self._next_term()
            if string in self._remaining_avoid_terms:
                self._remaining_avoid_terms.remove(string)
            else:
                return string

    def _next_term(self):
        string = ""
        for i in self._pointers:
            string += self._possible_values[i]
        self._increment_pointers()
        return string

    def __repr__(self):
        return "[{}]: {{{}}}".format(
            self.__class__.__name__, ", ".join(self._possible_values))
