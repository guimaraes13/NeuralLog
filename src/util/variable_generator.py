"""
Generates logic variable names.
"""
from collections import deque


class VariableGenerator:
    """
    Generate unique logic variable names.
    """

    def __init__(self):
        """
        Creates a logic variable generator.
        """
        self._possible_values = list(map(lambda x: chr(65 + x), range(26)))
        self._max_index = len(self._possible_values) - 1
        self._pointers = deque([0])

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
        string = ""
        for i in self._pointers:
            string += self._possible_values[i]
        self._increment_pointers()
        return string

    def __repr__(self):
        return "[{}]: {{{}}}".format(
            self.__class__.__name__, ", ".join(self._possible_values))
