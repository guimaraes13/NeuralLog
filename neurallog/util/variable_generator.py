"""
Generates logic variable names.
"""
import re
from collections import deque
from typing import Set, Iterator, Iterable, Dict, Sequence, List

from neurallog.language.language import Term, Predicate

GENERATOR_PATTERN = re.compile(r"[A-Z]+")


class VariableGenerator(Iterator[str], Iterable[str]):
    """
    Generate unique logic variable names.
    """

    def __init__(self, avoid_terms=()):
        """
        Creates a logic variable generator.

        :param avoid_terms: a collection of terms to be avoided by the
        variable generator. The generator will not generate terms in the
        avoided collection.
        :type avoid_terms: Iterator[str or Term] or Collections[str or Term]
        """
        self._possible_values = list(map(lambda x: chr(65 + x), range(26)))
        self._max_index = len(self._possible_values) - 1
        self._pointers = deque([0])
        self._all_avoid_terms: Set[str] = set(avoid_terms)
        self._remaining_avoid_terms = set()
        self.append_avoid_terms(avoid_terms)

    def append_avoid_terms(self, avoid_terms):
        """
        Appends the terms to avoid.

        :param avoid_terms: a collection of terms to be avoided by the
        variable generator. The generator will not generate terms in the
        avoided collection.
        :type avoid_terms: Iterator[str or Term]
        """
        current_term = self._get_current_term()
        for term in avoid_terms:
            if isinstance(term, Term):
                term = term.value
            if GENERATOR_PATTERN.fullmatch(term) is None \
                    or (len(term), term) < (len(current_term), current_term):
                continue
            self._remaining_avoid_terms.add(term)

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
        string = self._get_current_term()
        self._increment_pointers()
        return string

    def _get_current_term(self):
        string = ""
        for i in self._pointers:
            string += self._possible_values[i]
        return string

    def __repr__(self):
        return "[{}]: {{{}}}".format(
            self.__class__.__name__, ", ".join(self._possible_values))


class PredicateGenerator(Iterator[str], Iterable[str]):
    """
    Generate unique predicate names.
    """

    def __init__(self, avoid_terms=(), name_format="f{}"):
        """
        Creates a predicate name generator.

        :param avoid_terms: a collection of terms to be avoided by the
        variable generator. The generator will not generate terms in the
        avoided collection.
        :type avoid_terms: Iterator[str or Term] or Collections[str or Term]
        :param name_format: the name format
        :type name_format: str
        """
        self.name_format = name_format
        self._name_regex = \
            re.compile(self.name_format.format(r"([0-9]|[1-9][0-9]+)"))
        self._current_index = 0
        self._all_avoid_terms: Set[str] = set(avoid_terms)
        self._remaining_avoid_terms = set()
        self.append_avoid_terms(avoid_terms)

    def append_avoid_terms(self, avoid_terms):
        """
        Appends the terms to avoid.

        :param avoid_terms: a collection of terms to be avoided by the
        variable generator. The generator will not generate terms in the
        avoided collection.
        :type avoid_terms: Iterator[str or Term]
        """
        for term in avoid_terms:
            if isinstance(term, Predicate):
                term = term.name
            match = self._name_regex.fullmatch(term)
            if match is not None \
                    and int(match.groups()[0]) > self._current_index:
                self._remaining_avoid_terms.add(term)

    # noinspection PyMethodMayBeStatic
    def clean_copy(self):
        """
        Returns a clean copy of the variable generator class.

        :return: A clean copy of this class
        :rtype: VariableGenerator
        """
        return PredicateGenerator(self._all_avoid_terms)

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
        string = self.name_format.format(self._current_index)
        self._current_index += 1
        return string

    def __repr__(self):
        return "[{}]: {{{}}}".format(self.__class__.__name__, self.name_format)


class PermutationGenerator(Iterator[Dict[Predicate, str]],
                           Iterable[Dict[Predicate, str]]):
    """
    Creates the permutations of the predicates.
    """

    def __init__(self, predicate_terms, permutation_terms):
        """
        Creates a permutation generator.

        :param predicate_terms: a list of predicates
        :type predicate_terms: List[Predicate]
        :param permutation_terms: a list of sequences with the possible
        substitution for each predicate from predicate terms
        :type permutation_terms: List[Sequence[str]]
        """
        self.predicate_terms = predicate_terms
        self.permutation_terms = permutation_terms
        self._maximum_indices = \
            list(map(lambda x: len(x) - 1, permutation_terms))
        self._pointers = [0] * len(permutation_terms)
        self._last_index = False

    # noinspection PyMethodMayBeStatic
    def clean_copy(self):
        """
        Returns a clean copy of the variable generator class.

        :return: A clean copy of this class
        :rtype: VariableGenerator
        """
        return \
            PermutationGenerator(self.predicate_terms, self.permutation_terms)

    def _increment_pointers(self):
        next_step = 1
        for i in reversed(range(len(self._pointers))):
            if self._pointers[i] == self._maximum_indices[i]:
                self._pointers[i] = 0
                next_step = 1
            else:
                self._pointers[i] += next_step
                next_step = 0
                break
        if next_step > 0:
            self._last_index = True

    def __iter__(self):
        return self

    def __next__(self):
        if self._last_index:
            raise StopIteration
        result = self._get_current_term()
        self._increment_pointers()
        return result

    def _get_current_term(self):
        result = dict()
        for i in range(len(self._pointers)):
            result[self.predicate_terms[i]] = \
                self.permutation_terms[i][self._pointers[i]]
        return result

    def __repr__(self):
        return "[{}]: {{{}}}".format(
            self.__class__.__name__, ", ".join(
                map(lambda x: str(x), self.predicate_terms)))
