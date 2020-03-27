"""
Defines a NeuralLog Program.
"""
import collections
import logging
import re
import sys
from collections import OrderedDict, deque
from typing import TypeVar, MutableMapping, Dict, Any, List, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from src.language.language import Number, TermType, Predicate, Atom, \
    HornClause, Term, AtomClause, ClauseMalformedException, \
    PredicateTypeError, \
    UnsupportedMatrixRepresentation, Literal, \
    get_constant_from_string

ANY_PREDICATE_NAME = ":any:"
NO_EXAMPLE_SET = ":none:"

MAX_NUMBER_OF_ARGUMENTS = -1

PREDICATE_TYPE_MATCH = re.compile("\\$([a-zA-Z_-][a-zA-Z0-9_-]*)"
                                  "/([1-9][0-9]*)\\[([0-9]+)\\](\\[.+\\])?")

KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.

logger = logging.getLogger()


def build_builtin_predicate():
    """
    Decorates the function to handle builtin predicates.

    :return: a function to registry the function
    :rtype: function
    """
    commands = {}

    def decorator(predicate):
        """
        Returns a function to register the command with the name of
        the predicate.

        :param predicate: the predicate name
        :type predicate: str
        :return: a function to register the command
        :rtype: function
        """

        def registry(func):
            """
            Registries the function as a command to handle the builtin
            predicate.

            :param func: the function to be registered.
            :type func: function
            :return: the registered function
            :rtype: function
            """
            commands[predicate] = func
            return func

        return registry

    decorator.functions = commands
    return decorator


def get_term_type(term):
    """
    Returns the type of the term.

    :param term: the term
    :type term: Term
    :return: the type of the term
    :rtype: TermType
    """
    if isinstance(term, Number):
        return TermType(False, True)
    elif term.is_constant():
        return TermType(False, False)
    else:
        return TermType(True, None)


def get_predicate_type(atom):
    """
    Creates a tuple containing the type of each term in the atom.

    :param atom: the atom
    :type atom: Atom
    :return: a tuple containing the type of each predicate in the atom
    :rtype: tuple[TermType]
    """
    types = []
    for term in atom.terms:
        types.append(get_term_type(term))
    return tuple(types)


def get_updated_type(first, second):
    """
    Gets the update types from the two collections.

    :param first: the types of the first predicate
    :type first: list[TermType] or tuple[TermType]
    :param second: the types of the second predicate
    :type second: list[TermType] or tuple[TermType]
    :return: the update types, if possible; otherwise, none
    :rtype: tuple[TermType]
    """
    if len(first) != len(second):
        return None
    updated_types = []
    for i in range(len(first)):
        updated_type = first[i].update_type(second[i])
        if updated_type is None:
            return None
        updated_types.append(updated_type)

    return updated_types


def get_predicate_from_string(string):
    """
    Gets a predicate from the string
    :param string: the string
    :type string: str
    :return: the predicate
    :rtype: Predicate
    """
    if "/" in string:
        try:
            index = string.rfind("/")
            arity = int(string[index + 1:])
            return Predicate(string[:index], arity)
        except ValueError:
            pass

    return Predicate(string, -1)


def print_neural_log_program(program, writer=sys.stdout):
    """
    Prints the NeuralLogProgram to the writer.

    :param program: the program
    :type program: NeuralLogProgram
    :param writer: the writer. Default is to print to the standard output
    """
    key = lambda x: (x.arity, x.name)
    for predicate in sorted(program.facts_by_predicate.keys(), key=key):
        for item in program.facts_by_predicate[predicate].values():
            print(AtomClause(item), file=writer)
        print(file=writer)

    print(file=writer)
    for predicate in sorted(program.clauses_by_predicate.keys(), key=key):
        for item in program.clauses_by_predicate[predicate]:
            print(item, file=writer)
        print(file=writer)


# noinspection DuplicatedCode
def get_value_from_parameter(names, parameters):
    """
    Gets the value from the `parameters` dict.

    :param names: the names of the value
    :type names: list[str]
    :param parameters: the parameters dict
    :type parameters: [str, dict or str]
    :return: the value of the parameter
    :rtype: str or None
    """
    value = None
    for key in names:
        value = parameters.get(key, None)
        if not isinstance(value, dict):
            break
    return value


def _convert_to_bool(value):
    """
    If the value is `true` or `false`, return the boolean equivalent form of
    the value. This method is case insensitive.

    :param value: the string value
    :type value: str
    :return: the string value or the boolean value
    :rtype: str or bool
    """
    if isinstance(value, str):
        lower_value = value.lower()
        if lower_value == "true":
            value = True
        elif lower_value == "false":
            value = False
    return value


def _build_constant_dict(_iterable_constants):
    """
    Builds the dictionary of iterable constants.

    :param _iterable_constants: the iterable constants
    :type _iterable_constants: collection.Iterable[Term]
    :return: the dictionary of iterable constants
    :rtype: BiDict[int, Term]
    """
    count = 0
    dictionary = BiDict()
    for constant in sorted(_iterable_constants, key=lambda x: x.__str__()):
        dictionary[count] = constant
        count += 1
    return dictionary


def _join_sets_with_common_elements(sets):
    """
    Groups sets that has common elements and returns a list of disjoint
    sets.

    :param sets: the sets of elements
    :type sets: list[set]
    :return: the list of disjoint sets
    :rtype: list[set]
    """
    current_sets = list(sets)
    join_sets = list()  # type: List[set]
    while True:
        for elements_set in current_sets:
            not_joined = True
            for join_set in join_sets:
                if not join_set.isdisjoint(elements_set):
                    not_joined = False
                    join_set.update(elements_set)
            if not_joined:
                join_sets.append(elements_set)
        if len(join_sets) == len(current_sets):
            break
        current_sets = join_sets
        join_sets = list()

    return join_sets


class TooManyArguments(Exception):
    """
    Represents an exception raised by an atom with too many arguments.
    """

    def __init__(self, clause, found) -> None:
        """
        Creates a too many arguments exception.

        :param clause: the atom
        :type clause: Clause
        :param found: the number of arguments found
        :type found: int
        """
        super().__init__(
            "Too many arguments found for {} at line {} in file {}. "
            "Found {} arguments, the maximum number of arguments "
            "allowed is {}.".format(clause, clause.provenance.start_line,
                                    clause.provenance.filename, found,
                                    MAX_NUMBER_OF_ARGUMENTS))


class SimpleRulePathFinder:
    """
    Represents a rule graph
    """

    edges: Dict[Term, List[Literal]]
    loops: Dict[Term, List[Literal]]

    def __init__(self, clause):
        """
        Creates a rule graph.

        :param clause: the clause
        :type clause: HornClause
        """
        self.clause = clause
        self._build_graph()

    def _build_graph(self):
        """
        Builds the graph from the clause.

        :return: the graph
        :rtype: dict[Term, list[Literal]]
        """
        self.edges = dict()
        self.loops = dict()
        for literal in self.clause.body:
            if literal.arity() == 0:
                continue
            elif literal.arity() == 1 or literal.terms[0] == literal.terms[-1]:
                self.loops.setdefault(literal.terms[0], []).append(literal)
            else:
                self.edges.setdefault(literal.terms[0], []).append(literal)
                self.edges.setdefault(literal.terms[-1], []).append(literal)

    def find_clause_paths(self, inverted=False):
        """
        Finds the paths in the clause.

        :param inverted: if `True`, creates the paths for the inverted rule;
        this is, the rule in the format (output, input). If `False`,
        creates the path for the standard (input, output) rule format.
        :type inverted: bool
        :return: the completed paths between the terms of the clause and the
        remaining grounded literals
        :rtype: (List[SimpleRulePath], List[Literal])
        """
        # Defining variables
        source = self.clause.head.terms[0]
        destination = self.clause.head.terms[-1]
        if inverted:
            source, destination = destination, source

        visited_literals = set()
        paths = self.find_forward_paths(source, destination, visited_literals)

        if self.clause.head.predicate.arity > 1 and \
                not visited_literals.issuperset(self.clause.body):
            # Finding backward paths
            source, destination = destination, source
            backward_paths = self.find_forward_paths(source, destination,
                                                     visited_literals)
            path_set = set(paths)
            for backward_path in backward_paths:
                reversed_path = backward_path.reverse()
                if reversed_path not in path_set:
                    path_set.add(reversed_path)
                    paths.append(reversed_path)

        ground_literals = self.get_disconnected_literals(visited_literals)

        return paths, ground_literals

    def get_disconnected_literals(self, connected_literals):
        """
        Gets the literals from the `clause` which are disconnected from the
        source and destination variables but are grounded.

        :param connected_literals: the set of connected literals
        :type connected_literals: Set[Literal]
        :return: the list of disconnected literals
        :rtype: List[Literal]
        """
        ground_literals = []
        for literal in self.clause.body:
            if literal in connected_literals or not literal.is_grounded():
                continue
            ground_literals.append(literal)
        return ground_literals

    # noinspection DuplicatedCode
    def find_forward_paths(self, source, destination, visited_literals):
        """
        Finds all forward paths from `source` to `destination` by using the
        literals in the clause.
        If the destination cannot be reached, it includes a special `any`
        predicated to connect the path.

        :param source: the source of the paths
        :type source: Term
        :param destination: the destination of the paths
        :type destination: Term
        :param visited_literals: the set of visited literals
        :type visited_literals: Set[Literal]
        :return: the completed forward paths between source and destination
        :rtype: List[SimpleRulePath]
        """
        partial_paths = deque()  # type: deque[SimpleRulePath]

        initial_path = self.build_initial_path(source, visited_literals)
        for literal in self.edges.get(source, []):
            if literal in visited_literals:
                continue
            new_path = initial_path.new_path_with_item(literal)
            if new_path is not None:
                visited_literals.add(literal)
                if new_path.path_end() != destination:
                    for loop in self.loops.get(new_path.path_end(), []):
                        new_path.append(loop)
                        visited_literals.add(loop)
                partial_paths.append(new_path)
        if len(partial_paths) == 0:
            partial_paths.append(initial_path)

        return self.find_paths(partial_paths, destination, visited_literals)

    @staticmethod
    def complete_path_with_any(dead_end_paths, destination):
        """
        Completes the path by appending the special `any` predicate between the
        end of the path and the destination.

        :param dead_end_paths: the paths to be completed
        :type dead_end_paths: collections.Iterable[SimpleRulePath]
        :param destination: the destination
        :type destination: Term
        :return: the completed paths
        :rtype: List[SimpleRulePath]
        """
        completed_paths = []
        for path in dead_end_paths:
            any_literal = Literal(Atom(ANY_PREDICATE_NAME,
                                       path.path_end(), destination))
            path.append(any_literal)
            completed_paths.append(path)
        return completed_paths

    def build_initial_path(self, source, visited_literals):
        """
        Builds a path with its initial literals, if any.

        :param source: the source of the path
        :type source: Term
        :param visited_literals: the set of visited literals
        :type visited_literals: set[Literal]
        :return: the path or `None`
        :rtype: SimpleRulePath
        """
        loop_literals = self.loops.get(source, [])
        path = SimpleRulePath(source, loop_literals)
        visited_literals.update(path.literals)
        return path

    # noinspection DuplicatedCode
    def find_paths(self, partial_paths, destination, visited_literals):
        """
        Finds the paths from `partial_paths` to `destination` by appending the
        literals from clause.

        :param partial_paths: The initial partial paths
        :type partial_paths: deque[SimpleRulePath]
        :param destination: the destination term
        :type destination: Term
        :param visited_literals: the visited literals
        :type visited_literals: Set[Literal]
        :return: the completed paths
        :rtype: List[SimpleRulePath]
        """
        completed_paths = deque()  # type: deque[SimpleRulePath]
        while len(partial_paths) > 0:
            size = len(partial_paths)
            for i in range(size):
                path = partial_paths.popleft()
                path_end = path.path_end()

                if path_end == destination:
                    if path.source != destination:
                        for loop in self.loops.get(destination, []):
                            path.append(loop)
                            visited_literals.add(loop)
                    completed_paths.append(path)
                    continue

                possible_edges = self.edges.get(path_end, [])
                not_added_path = True
                for literal in possible_edges:
                    new_path = path.new_path_with_item(literal)
                    if new_path is None:
                        continue
                    end = new_path.path_end()
                    if end != destination:
                        for loop in self.loops.get(end, []):
                            new_path.append(loop)
                            visited_literals.add(loop)
                    partial_paths.append(new_path)
                    # noinspection PyTypeChecker
                    visited_literals.add(literal)
                    not_added_path = False

                if not_added_path:
                    if path.source == destination:
                        completed_paths.append(path)
                    else:
                        any_literal = Literal(Atom(
                            ANY_PREDICATE_NAME, path.path_end(), destination))
                        path.append(any_literal)
                        partial_paths.append(path)
        return completed_paths

    def __str__(self):
        return self.clause.__str__()

    def __repr__(self):
        return self.clause.__repr__()


class SimpleRulePath:
    """
    Represents a rule path.
    """

    source: Term
    "The source term"

    path: List[Literal]
    "The path of literals"

    literals: Set[Literal]
    "The set of literals in the path"

    terms: Set[Term]
    "The set of all terms in the path"

    inverted: List[bool]
    """It is True if the correspondent literal is inverted;
    it is false, otherwise"""

    def __init__(self, source, path=()):
        """
        Initializes a path.

        :param source: the source term
        :type source: Term
        :param path: the path
        :type path: collections.Iterable[Literal]
        """
        self.source = source
        self.path = list()
        self.literals = set()
        self.terms = set()
        self.inverted = list()
        for literal in path:
            self.append(literal)

    def path_end(self):
        """
        Gets the term at the end of the path.

        :return: the term at the end of the path
        :rtype: Term
        """
        if len(self.path) == 0:
            return self.source
        return self.path[-1].terms[0 if self.inverted[-1] else -1]

    def append(self, item):
        """
        Appends the item to the end of the path, if it is not in the path yet.

        :param item: the item
        :type item: Literal
        :return: True, if the item has been appended to the path; False,
        otherwise.
        :rtype: bool
        """
        if item is None:
            return False

        if item.arity() == 1 or item.terms[0] == item.terms[-1]:
            output_variable = item.terms[0]
            last_inverted = False
        else:
            if item.terms[0] == self.path_end():
                output_variable = item.terms[-1]
                last_inverted = False
            else:
                # inverted literal case
                output_variable = item.terms[0]
                if item.predicate.name == ANY_PREDICATE_NAME:
                    # if the literal is any, reverse the terms and make it
                    # not inverted. Since any^{-1} == any
                    item = Literal(
                        Atom(item.predicate, *list(reversed(item.terms)),
                             weight=item.weight), negated=item.negated)
                    last_inverted = False
                else:
                    last_inverted = True

        # Detects if there is a loop in the path
        if item.arity() != 1 and \
                item.terms[0] != item.terms[-1] and \
                output_variable in self.terms:
            return False

        self.path.append(item)
        self.literals.add(item)
        self.terms.update(item.terms)
        self.inverted.append(last_inverted)
        return True

    def new_path_with_item(self, item):
        """
        Creates a new path with `item` at the end.

        :param item: the item
        :type item: Literal or None
        :return: the new path, if its is possible to append the item; None,
        otherwise
        :rtype: SimpleRulePath or None
        """
        path = SimpleRulePath(self.source, self.path)
        # path = self._copy()
        return path if path.append(item) else None

    def reverse(self):
        """
        Gets a reverse path.

        :return: the reverse path
        :rtype: SimpleRulePath
        """
        source = self.path[-1].terms[0 if self.inverted[-1] else -1]
        return SimpleRulePath(source, reversed(self.path))

    def __getitem__(self, item):
        return self.path.__getitem__(item)

    def __len__(self):
        return self.path.__len__()

    def __str__(self):
        message = []
        for i in range(0, len(self.path)):
            prefix = self.path[i].predicate.name
            iterator = list(map(lambda x: x.value, self.path[i].terms))
            if self.inverted[i]:
                prefix += "^{-1}"
                iterator = reversed(iterator)

            prefix += "("
            prefix += ", ".join(iterator)
            prefix += ")"
            message.append(prefix)

        return ", ".join(message)

    __repr__ = __str__

    def __hash__(self):
        return hash((self.source, tuple(self.path)))

    def __eq__(self, other):
        if not isinstance(other, SimpleRulePath):
            return False
        return self.source == other.source, self.path == other.path


class BiDict(dict, MutableMapping[KT, VT]):
    """
    A bidirectional dictionary.
    The inverse dictionary is in the variable inverse.
    """

    def __init__(self, *args, **kwargs):
        super(BiDict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key in self.keys():
            self.inverse[self[key]] = key

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(BiDict, self).__setitem__(key, value)
        self.inverse[value] = key

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(BiDict, self).__delitem__(key)


class NeuralLogProgram:
    """
    Represents a NeuralLog language.
    """

    BUILTIN_PREDICATES = {
        "example": [Predicate("example", -1)],
        "learn": [Predicate("learn", 1)],
        "set_parameter": [Predicate("set_parameter", -1)],
        "set_predicate_parameter": [Predicate("set_predicate_parameter", -1)],
        "mega_example": [Predicate("mega_example", -1)]
    }

    builtin = build_builtin_predicate()

    facts_by_predicate: Dict[Predicate, Dict[Any, Atom]]
    """The facts. The values of this variable are dictionaries where the key 
    are the predicate and a tuple of the terms and the values are the atoms 
    itself. It was done this way in order to collapse different definitions 
    of the same atom with different weights, in this way, only the last 
    definition will be considered"""

    examples: Dict[str, Dict[Predicate, Dict[Any, Atom]]]
    """The examples. The values of this variable are dictionaries where the key 
    are the predicate and a tuple of the terms and the values are the atoms 
    itself. It was done this way in order to collapse different definitions 
    of the same atom with different weights, in this way, only the last 
    definition will be considered"""

    mega_examples: Dict[str, Dict[Any, Dict[Predicate, List[Atom]]]]
    """The mega examples. The values of this variable are dictionaries where 
    the key are the predicate and a tuple of the terms and the values are the 
    atoms itself. It was done this way in order to collapse different 
    definitions of the same atom with different weights, in this way, only the 
    last definition will be considered"""

    clauses_by_predicate: Dict[Predicate, List[HornClause]]
    "The clauses by predicate"

    constants: Set[Term]
    "All the constants"

    iterable_constants_per_term: Dict[Tuple[Predicate, int], BiDict[int, Term]]
    "The iterable constants per (predicate / term position)"

    predicates: Dict[Predicate, Tuple[TermType]]
    "All the predicates and their types"

    logic_predicates: Set[Predicate]
    "The logic predicates"

    functional_predicates: Set[Predicate]
    "The functional predicates"

    trainable_predicates: Set[Predicate]
    "The trainable predicates"

    parameters: Dict[Any, Any]
    "A dictionary with the parameters defined in the program"

    # predicate_functions: Dict[Any, Any] = dict()
    # "A dictionary with the functions defined for predicates in the program"

    _predicate_parameters_to_add: List[Atom]
    _parameters_to_add: List[Atom]

    def __init__(self):
        """
        Creates a NeuralLog Program.
        """
        self.facts_by_predicate = dict()
        self.examples = OrderedDict()
        self.mega_examples = OrderedDict()
        self.clauses_by_predicate = dict()
        self.constants = set()
        self.predicates = dict()
        self.logic_predicates = set()
        self.functional_predicates = set()
        self.trainable_predicates = set()
        self.parameters = dict()
        self._predicate_parameters_to_add = list()
        self._parameters_to_add = list()
        self._last_atom_for_predicate: Dict[Predicate, Atom] = dict()
        # self.add_clauses(clauses)
        # del self._last_atom_for_predicate
        # self.build_program()

    def build_program(self):
        """
        Builds the program after all the clauses had been added.
        """
        self._expand_clauses()
        self._add_specific_parameter("avoid_constant")
        self._get_constants()
        self._add_parameters()

    def add_clauses(self, clauses, *args, **kwargs):
        """
        Splits the clauses from the facts and create a map of the type of the
        predicates.

        :param clauses: the clauses
        :type clauses: collections.iterable[Clause]
        :raise ClauseMalformedException case the clause is malformed
        """
        for clause in clauses:
            if isinstance(clause, AtomClause):
                if self._is_builtin_predicate(clause.atom.predicate):
                    self._process_builtin_clause(clause, *args, **kwargs)
                    continue

            if isinstance(clause, AtomClause) and clause.is_grounded():
                self.add_fact(clause.atom, True)
            elif isinstance(clause, HornClause):
                self._add_predicate(clause.head)
                self.logic_predicates.add(clause.head.predicate)
                for atom in clause.body:
                    self._add_predicate(atom)
                self.clauses_by_predicate.setdefault(clause.head.predicate,
                                                     list()).append(clause)
            else:
                raise ClauseMalformedException(clause)

    def _expand_clauses(self):
        expanded_trainable = set()
        for predicate in self.predicates.keys():
            for trainable in self.trainable_predicates:
                if trainable.equivalent(predicate):
                    expanded_trainable.add(predicate)
                    self.logic_predicates.add(predicate)
        self.trainable_predicates = expanded_trainable
        # noinspection PyTypeChecker
        self.functional_predicates = \
            self.predicates.keys() - self.logic_predicates

    def add_fact(self, atom, report_replacement=False):
        """
        Adds the atom to the program.

        :param atom: the atom
        :type atom: Atom
        :param report_replacement: if `True` logs a warning if the atom
        already exists in the program.
        :type report_replacement: bool
        """
        self._add_predicate(atom)
        fact_dict = self.facts_by_predicate.setdefault(
            atom.predicate, OrderedDict())
        old_atom = fact_dict.get(atom.simple_key(), None)
        if report_replacement and old_atom is not None:
            if old_atom.provenance is not None and atom.provenance is not None:
                logger.warning("Warning: atom %s defined at line %d "
                               "replaced by Atom %s defined at line %d.",
                               old_atom, old_atom.provenance.start_line,
                               atom, atom.provenance.start_line)
            elif atom.provenance is None:
                logger.warning("Warning: atom %s defined at line %d "
                               "replaced by Atom %s.",
                               old_atom, old_atom.provenance.start_line, atom)
            elif old_atom.provenance is None:
                logger.warning("Warning: atom %s replaced by Atom %s defined at"
                               " line %d.", old_atom,
                               atom, atom.provenance.start_line)
            else:
                logger.warning("Warning: atom %s replaced by Atom %s",
                               old_atom, atom)

        fact_dict[atom.simple_key()] = atom
        self.logic_predicates.add(atom.predicate)

    def _add_predicate(self, atom):
        """
        Add the predicate of the atom, and its type, to the map of predicates
        while creating the types of the terms.

        :param atom: the atom
        :type atom: Atom
        :raise PredicateTypeError case a predicate violated the type
        expressed before by another atom of the same predicate
        :raise TooManyArguments if the atom has more than 2 arguments
        """
        atom_predicate = atom.predicate
        if 0 < MAX_NUMBER_OF_ARGUMENTS < atom_predicate.arity:
            raise TooManyArguments(atom, atom_predicate.arity)
        # atom.context = None
        types = get_predicate_type(atom)
        if atom_predicate not in self.predicates:
            self.predicates[atom_predicate] = types
        else:
            updated_types = get_updated_type(types,
                                             self.predicates[atom_predicate])
            if updated_types is None:
                raise PredicateTypeError(atom,
                                         self._last_atom_for_predicate[
                                             atom_predicate])
            self.predicates[atom_predicate] = updated_types
        self._last_atom_for_predicate[atom_predicate] = atom

    def _get_constants(self):
        """
        Gets the constants from the clauses.
        """
        grouped_types = self._get_grouped_types()
        _iterable_constants_per_term = dict()
        for type_terms in grouped_types:
            iterable_constant = set()
            for predicate_term in type_terms:
                _iterable_constants_per_term[predicate_term] = iterable_constant

        self._get_constants_from_facts(_iterable_constants_per_term)
        self._get_constants_from_examples(_iterable_constants_per_term)
        self._get_constants_from_mega_examples(_iterable_constants_per_term)
        self._get_constants_from_clauses(_iterable_constants_per_term)
        self.iterable_constants_per_term = dict()
        for type_terms in grouped_types:
            value = _iterable_constants_per_term[list(type_terms)[0]]
            dictionary = _build_constant_dict(value)
            for predicate_term in type_terms:
                self.iterable_constants_per_term[predicate_term] = dictionary

    def _get_constants_from_facts(self, _constants_per_term):
        """
        Gets the constants from the facts

        :param _constants_per_term: the dictionary of iterable constants per
        atom term
        :type _constants_per_term: dict[tuple[Predicate, int], set[Term]]
        """
        for facts in self.facts_by_predicate.values():
            for fact in facts.values():
                self._get_constants_from_atom(fact, _constants_per_term)
                fact.provenance = None

    def _get_constants_from_examples(self, _constants_per_term):
        """
        Gets the constants from the examples

        :param _constants_per_term: the dictionary of iterable constants per
        atom term
        :type _constants_per_term: dict[tuple[Predicate, int], set[Term]]
        """
        for sets in self.examples.values():
            for predicate, examples in sets.items():
                indices = self._get_indices_to_add(predicate)
                for example in examples.values():
                    self._get_constants_from_atom(
                        example, _constants_per_term,
                        is_example=True, indices=indices)
                    example.provenance = None

    def _get_constants_from_mega_examples(self, _constants_per_term):
        """
        Gets the constants from the mega examples

        :param _constants_per_term: the dictionary of iterable constants per
        atom term
        :type _constants_per_term: dict[tuple[Predicate, int], set[Term]]
        """
        for sets in self.mega_examples.values():
            for id_sets in sets.values():
                for pred, examples in id_sets.items():
                    indices = self._get_indices_to_add(pred)
                    for example in examples:
                        self._get_constants_from_atom(
                            example, _constants_per_term,
                            is_example=True, indices=indices)
                        example.provenance = None

    def _get_indices_to_add(self, predicate):
        """
        Gets the indices to add by the predicate
        :param predicate: the predicate
        :type predicate: Predicate
        :return: the indices to add
        :rtype: set[int]
        """
        avoid = self.get_parameter_value("avoid_constant", predicate)
        indices = None
        if avoid is not None:
            if isinstance(avoid, dict):
                avoid = set(avoid.values())
            else:
                avoid = {avoid}
            indices = set(range(predicate.arity)) - avoid
        return indices

    def _get_constants_from_clauses(self, _constants_per_term):
        """
        Gets the constants from the clauses

        :param _constants_per_term: the dictionary of iterable constants per
        atom term
        :type _constants_per_term: dict[tuple[Predicate, int], set[Term]]
        """
        for clauses in self.clauses_by_predicate.values():
            for clause in clauses:
                self._get_constants_from_atom(
                    clause.head, _constants_per_term)
                clause.head.provenance = None
                for literal in clause.body:
                    self._get_constants_from_atom(
                        literal, _constants_per_term)
                    literal.provenance = None

    def _get_grouped_types(self):
        """
        Gets the groups of types of each predicate term position.
        :return: the dictionary of iterable constants per
        (predicate, term position)
        :rtype: list[set[tuple[Predicate, int]]]
        """
        type_sets = list()
        for clauses in self.clauses_by_predicate.values():
            for clause in clauses:
                term_types = dict()
                self._find_term_type(clause.head, term_types)
                for literal in clause.body:
                    self._find_term_type(literal, term_types)
                type_sets += list(term_types.values())
        return _join_sets_with_common_elements(type_sets)

    def _find_term_type(self, atom, term_types):
        """
        Finds the types of the terms in the atom.

        For each term of the atom, it adds the term type (composed of the
        predicate and the position of the term) to the set of types where the
        term appears.

        :param atom: the atom
        :type atom: Atom
        :param term_types: a dictionary with the set of types where the
        term appears
        :type term_types: dict[Term, set[Tuple[Predicate, int]]]
        :return: a dictionary with the set of types where the
        term appears
        :rtype: dict[Term, set[Tuple[Predicate, int]]]
        """
        numeric_terms = []
        not_numeric_terms = []
        for i in range(atom.arity()):
            if self.predicates[atom.predicate][i].number:
                numeric_terms.append(i)
            else:
                not_numeric_terms.append(i)
                term_types.setdefault(atom.terms[i],
                                      set()).add((atom.predicate, i))
        for i in numeric_terms:
            for j in not_numeric_terms:
                term_types.setdefault(atom.terms[i],
                                      set()).add((atom.predicate, j))

        return term_types

    def _get_constants_from_atom(self, atom, iterable_constant_per_term,
                                 is_example=False, indices=None):
        """
        Gets the constants from an atom.

        :param atom: the atom
        :type atom: Atom
        :param iterable_constant_per_term: the iterable constants per
        (predicate, term position)
        :type iterable_constant_per_term: dict[tuple[Predicate, int], set[Term]]
        :param is_example: if the atom is an example
        :type is_example: bool
        """
        types = self.predicates[atom.predicate]
        if indices is None:
            indices = range(len(types))

        for i in indices:
            if not atom[i].is_constant() or types[i].number:
                continue
            self.constants.add(atom[i])
            if is_example or types[i].variable:
                iterable_constant_per_term[(atom.predicate, i)].add(atom[i])

    def get_constant_size(self, predicate, index):
        """
        Gets the constant size of the predicate term.

        :param predicate: the predicate
        :type predicate: Predicate
        :param index: the index of the term
        :type index: int
        :return: the constant size of the predicate term
        :rtype: int
        """
        return len(self.iterable_constants_per_term.get((predicate, index),
                                                        dict()))

    def is_iterable_constant(self, atom, term_index):
        """
        Checks if the term of the atom is an iterable constant.

        :param atom: the atom
        :type atom: Atom
        :param term_index: the index of the term
        :type term_index: int
        :return: `True`, if the term is an iterable constant; otherwise, `False`
        :rtype: bool
        """
        dictionary = self.iterable_constants_per_term.get(
            (atom.predicate, term_index), None)
        if dictionary is not None:
            return atom.terms[term_index] in dictionary.inverse
        return False

    def get_constant_by_index(self, predicate, term_index, constant_index):
        """
        Gets the constant name by the index.

        :param predicate: the predicate
        :type predicate: Predicate
        :param term_index: the index of the predicate term
        :type term_index: int
        :param constant_index: the constant index
        :type constant_index: int or ndarray[int]
        :return: the constant
        :rtype: Term
        """
        if term_index < 0:
            term_index = predicate.arity + term_index
        # noinspection PyTypeChecker
        return self.iterable_constants_per_term[
            (predicate, term_index)][constant_index]

    def get_index_of_constant(self, predicate, term_index, constant):
        """
        Gets the index of the constant.

        :param predicate: the predicate
        :type predicate: Predicate
        :param term_index: the index of the predicate term
        :type term_index: int
        :param constant: the constant
        :type constant: Term
        :return: the index of the constant
        :rtype: int
        """
        # noinspection PyTypeChecker
        return self.iterable_constants_per_term[
            (predicate, term_index)].inverse.get(constant, None)

    def get_matrix_representation(self, predicate, mask=False):
        """
        Builds the matrix representation for the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :param mask: if True, instead of the weights, returns 1.0 if for the
        facts that appears in the knowledge base, even if its weight is 0.0;
        or 0.0 otherwise
        :type mask: bool
        :raise UnsupportedMatrixRepresentation in the case the predicate is
        not convertible to matrix form
        :return: the matrix representation of the data for the given predicate
        :rtype: csr_matrix or np.matrix or (csr_matrix, csr_matrix) or float
        """
        true_arity = self.get_true_arity(predicate)
        if true_arity == 0:
            if predicate.arity == 0:
                return self._propositional_matrix_representation(predicate,
                                                                 mask)
            else:
                return self._attribute_numeric_representation(predicate, mask)
        elif true_arity == 1:
            if predicate.arity == 1:
                return self._relational_matrix_representation(predicate, mask)
            elif predicate.arity == 2:
                return self._attribute_matrix_representation(predicate, mask)
        elif true_arity == 2:
            return self._relational_matrix_representation(predicate, mask)
        raise UnsupportedMatrixRepresentation(predicate)

    def get_true_arity(self, predicate):
        """
        Returns the true arity of the predicate. The number of terms that are
        not number.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the true arity
        :rtype: int
        """
        return sum(1 for i in self.predicates[predicate] if not i.number)

    def _propositional_matrix_representation(self, predicate, mask=False):
        """
        Builds the numeric representation for the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :param mask: if True, instead of the weights, returns 1.0 if for the
        facts that appears in the knowledge base, even if its weight is 0.0;
        or 0.0 otherwise
        :type mask: bool
        :return: the float representation of the data, the weight of the
        fact and the attribute
        :rtype: (float, float)
        """
        facts = self.facts_by_predicate.get(predicate, None)
        if facts is None or len(facts) == 0:
            return 0.0
        else:
            return 1.0 if mask else list(facts.values())[0].weight

    def _attribute_numeric_representation(self, predicate, mask=False):
        """
        Builds the attribute numeric representation for the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :param mask: if True, instead of the weights, returns 1.0 if for the
        facts that appears in the knowledge base, even if its weight is 0.0;
        or 0.0 otherwise
        :type mask: bool
        :return: the float representation of the data, the weight of the
        fact and the attribute
        :rtype: (float, float)
        """
        facts = self.facts_by_predicate.get(predicate, None)
        if facts is None or len(facts) == 0:
            if mask:
                return 0.0
            return 0.0, [0.0] * predicate.arity
        else:
            if mask:
                return 1.0
            fact = list(facts.values())[0]
            return fact.weight, [x.value for x in fact.terms]

    def _attribute_matrix_representation(self, predicate, mask=False):
        """
        Builds the attribute matrix representation for the binary predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :param mask: if True, instead of the weights, returns 1.0 if for the
        facts that appears in the knowledge base, even if its weight is 0.0;
        or 0.0 otherwise
        :type mask: bool
        :return: the matrix representation of the data, the weights of the
        facts and the attributes of the entities
        :rtype: (csr_matrix, csr_matrix)
        """
        attribute_index = 0 if self.predicates[predicate][0].number else 1
        entity_index = 1 - attribute_index
        weight_data = []
        attribute_data = []
        entity_indices = []
        for fact in self.facts_by_predicate.get(predicate, dict()).values():
            index = self._check_iterable_terms(fact)
            if index is None:
                continue
            weight_data.append(1.0 if mask else fact.weight)
            attribute_data.append(fact.terms[attribute_index].value)
            entity_indices.append(index[0])

        size = self.get_constant_size(predicate, entity_index)
        weights = csr_matrix((weight_data,
                              (entity_indices, [0] * len(weight_data))),
                             shape=(size, 1),
                             dtype=np.float32)
        if mask:
            return weights

        return (weights,
                csr_matrix((attribute_data,
                            (entity_indices, [0] * len(attribute_data))),
                           shape=(size, 1), dtype=np.float32))

    def _relational_matrix_representation(self, predicate, mask=False):
        """
        Builds the relational matrix representation for the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :param mask: if True, instead of the weights, returns 1.0 if for the
        facts that appears in the knowledge base, even if its weight is 0.0;
        or 0.0 otherwise
        :type mask: bool
        :return: the matrix representation of the data
        :rtype: csr_matrix
        """
        data = []
        ind = []
        term_range = list(range(predicate.arity))
        for _ in term_range:
            ind.append([])
        for fact in self.facts_by_predicate.get(predicate, dict()).values():
            indices = self._check_iterable_terms(fact)
            if indices is None:
                continue
            for i in term_range:
                ind[i].append(indices[i])
            data.append(1.0 if mask else fact.weight)

        if predicate.arity == 1:
            size = self.get_constant_size(predicate, 0)
            return csr_matrix((data, (ind[0], [0] * len(data))),
                              shape=(size, 1),
                              dtype=np.float32)

        shape = []
        for i in range(predicate.arity):
            shape.append(self.get_constant_size(predicate, i))

        return csr_matrix(
            (data, tuple(ind)), shape=tuple(shape), dtype=np.float32)

    def get_diagonal_matrix_representation(self, predicate, mask=False):
        """
        Builds the diagonal matrix representation for a binary predicate.

        :param predicate: the binary predicate
        :type predicate: Predicate
        :param mask: if True, instead of the weights, returns 1.0 if for the
        facts that appears in the knowledge base, even if its weight is 0.0;
        or 0.0 otherwise
        :type mask: bool
        :return: the matrix representation of the data
        :rtype: csr_matrix
        """
        data = []
        ind = [[], []]

        for fact in self.facts_by_predicate.get(predicate, dict()).values():
            indices = self._check_iterable_terms(fact)
            if indices is None:
                continue
            if indices[0] != indices[1]:
                continue
            ind[0].append(indices[0])
            ind[1].append(0)
            data.append(1.0 if mask else fact.weight)

        size = self.get_constant_size(predicate, 0)
        return csr_matrix((data, tuple(ind)), shape=(size, 1), dtype=np.float32)

    def _check_iterable_terms(self, atom):
        """
        Checks if all the terms of an atom are iterable constants. If they all
        are, it returns a list of their indices, otherwise, returns `None`.

        :param atom: the atom
        :type atom: Atom
        :return: a list with the indices of the terms, if they all are
        iterable constants; otherwise, returns `None`
        :rtype: list[int] or None
        """
        indices = []
        for i in range(len(atom.terms)):
            if isinstance(atom.terms[i], Number):
                continue
            index = self.get_index_of_constant(atom.predicate, i, atom.terms[i])
            if index is None:
                return None
            indices.append(index)

        return indices

    def get_vector_representation_with_constant(self, atom, mask=False):
        """
        Gets the vector representation for a binary atom with a constant and
        a variable.

        :param atom: the binary atom
        :type atom: Atom
        :param mask: if True, instead of the weights, returns 1.0 if for the
        facts that appears in the knowledge base, even if its weight is 0.0;
        or 0.0 otherwise
        :type mask: bool
        :raise UnsupportedMatrixRepresentation in the case the predicate is
        not convertible to matrix form
        :return: the vector representation
        :rtype: csr_matrix
        """
        predicate = atom.predicate
        if self.get_true_arity(predicate) != 2:
            raise UnsupportedMatrixRepresentation(atom)

        constant_index = 0 if atom.terms[0].is_constant() else 1
        variable_index = 1 - constant_index
        data = []
        ind = [[], []]

        for fact in self.facts_by_predicate.get(predicate, dict()).values():
            if fact.terms[constant_index] != atom.terms[constant_index]:
                continue
            index = self.iterable_constants_per_term[
                (predicate, variable_index)].inverse.get(
                fact.terms[variable_index], None)
            if index is None:
                continue
            ind[0].append(index)
            ind[1].append(0)
            data.append(1.0 if mask else fact.weight)

        size = self.get_constant_size(predicate, variable_index)
        return csr_matrix((data, tuple(ind)), shape=(size, 1), dtype=np.float32)

    def _is_builtin_predicate(self, predicate):
        """
        Checks if the predicate is a builtin predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: if the predicate is a builtin predicate
        :rtype: bool
        """
        predicates = self.BUILTIN_PREDICATES.get(predicate.name, [])
        for builtin in predicates:
            if builtin.equivalent(predicate):
                return True

        return False

    def _process_builtin_clause(self, clause, *args, **kwargs):
        """
        Process the builtin clause.

        :param clause: the clause
        :type clause: AtomClause
        """
        predicate = clause.atom.predicate
        # noinspection PyUnresolvedReferences
        self.builtin.functions[predicate.name](self, clause, *args, **kwargs)

    def get_parameter_value(self, name, predicate=None):
        """
        Gets the name of the initializer for the predicate.

        :param name: the parameter's name
        :type name: collections.Iterable[str] or str
        :param predicate: the predicate, or `None` for default parameters
        :type predicate: Predicate
        :return: the value of the parameter
        """
        if isinstance(name, str):
            name = [name]

        if predicate is None:
            return get_value_from_parameter(name, self.parameters)
        else:
            parameters = self.parameters.get(predicate, self.parameters)
            value = get_value_from_parameter(name, parameters)
            if value is None:
                return get_value_from_parameter(name, self.parameters)
            return value

    # noinspection PyUnusedLocal
    @builtin("example")
    def _example(self, example, *args, **kwargs):
        """
        Process the builtin `example` predicate.

        :param example: the example clause
        :type example: AtomClause
        """
        atom = Atom(
            example.atom.terms[0].value,
            weight=example.atom.weight,
            *example.atom.terms[1:],
            provenance=example.atom.provenance
        )

        example_set = kwargs.get("example_set", NO_EXAMPLE_SET)
        example_dict = self.examples.setdefault(example_set, OrderedDict())
        example_dict = example_dict.setdefault(atom.predicate, OrderedDict())
        key = atom.simple_key()
        old_atom = example_dict.get(key, None)
        if old_atom is not None:
            logger.warning("Warning: example %s defined in file %s at line "
                           "%d replaced by Example %s defined in file %s at "
                           "line %d.",
                           old_atom, old_atom.provenance.filename,
                           old_atom.provenance.start_line,
                           atom, atom.provenance.filename,
                           atom.provenance.start_line)
        example_dict[key] = atom

    # noinspection PyUnusedLocal,DuplicatedCode
    @builtin("mega_example")
    def _mega_example(self, example, *args, **kwargs):
        """
        Process the builtin `mega_example` predicate.

        :param example: the example clause
        :type example: AtomClause
        """
        example_id = example.atom.terms[0].value
        atom = Atom(
            example.atom.terms[1].value,
            weight=example.atom.weight,
            *example.atom.terms[2:],
            provenance=example.atom.provenance
        )

        # noinspection DuplicatedCode
        example_set = kwargs.get("example_set", NO_EXAMPLE_SET)
        example_dict = self.mega_examples.setdefault(example_set, OrderedDict())
        example_dict = example_dict.setdefault(example_id, OrderedDict())
        example_dict = example_dict.setdefault(atom.predicate, [])
        # key = atom.simple_key()
        # old_atom = example_dict.get(key, None)
        # if old_atom is not None:
        #     logger.warning("Warning: mega example %s defined in file %s at "
        #                    "line %d replaced by Example %s defined in file "
        #                    "%s at line %d.",
        #                    old_atom, old_atom.provenance.filename,
        #                    old_atom.provenance.start_line,
        #                    atom, atom.provenance.filename,
        #                    atom.provenance.start_line)
        example_dict.append(atom)

    # noinspection PyUnusedLocal
    @builtin("learn")
    def _learn_predicate(self, clause, *args, **kwargs):
        """
        Process the builtin `learn` predicate.

        :param clause: the learn clause
        :type clause: AtomClause
        """
        predicate = get_predicate_from_string(clause.atom.terms[0].get_name())
        self.trainable_predicates.add(predicate)

    # noinspection PyUnusedLocal
    @builtin("set_parameter")
    def _set_parameter(self, clause, *args, **kwargs):
        """
        Process the builtin `set_parameter` predicate.

        :param clause: the set parameter clause
        :type clause: AtomClause
        """
        if clause.atom.arity() < 2:
            return

        self._parameters_to_add.append(clause.atom)

    # noinspection PyUnusedLocal
    @builtin("set_predicate_parameter")
    def _set_predicate_parameter(self, clause, *args, **kwargs):
        """
        Process the builtin `set_predicate_parameter` predicate.

        :param clause: the set predicate parameter clause
        :type clause: AtomClause
        """
        if clause.atom.arity() < 3:
            return

        self._predicate_parameters_to_add.append(clause.atom)

    # noinspection PyUnusedLocal
    def _add_parameters(self):
        for parameter in self._parameters_to_add:
            # parameter.terms[0].value
            self._set_parameter_to_dict(parameter)
        for parameter in self._predicate_parameters_to_add:
            self._set_predicate_parameter_to_dict(parameter)
        for parameter in DEFAULT_PARAMETERS:
            key, value = parameter[0], parameter[1]
            self.parameters.setdefault(key, value)

    # noinspection PyUnusedLocal
    def _add_specific_parameter(self, parameter_name):
        for parameter in self._parameters_to_add:
            if parameter.terms[0].value != parameter_name:
                continue
            self._set_parameter_to_dict(parameter)
        for parameter in self._predicate_parameters_to_add:
            if parameter.terms[1].value != parameter_name:
                continue
            self._set_predicate_parameter_to_dict(parameter)
        for parameter in DEFAULT_PARAMETERS:
            key, value = parameter[0], parameter[1]
            if key != parameter_name:
                continue
            self.parameters.setdefault(key, value)

    def _set_parameter_to_dict(self, atom):
        arity = atom.arity()
        parameter_dict = self.parameters
        for i in range(arity - 2):
            parameter_dict = parameter_dict.setdefault(
                atom.terms[i].value, dict())
        value = self._convert_value(atom.terms[-1].value)
        parameter_dict[atom.terms[-2].value] = value

    def _set_predicate_parameter_to_dict(self, atom):
        """
        Sets the atom parameter to `parameters` dictionary.

        :param atom: the atom
        :type atom: Atom
        """
        arity = atom.arity()

        parameter_dict = self.parameters
        predicate = get_predicate_from_string(atom.terms[0].value)
        parameter_dict = parameter_dict.setdefault(predicate, dict())
        for i in range(1, arity - 2):
            parameter_dict = parameter_dict.setdefault(atom.terms[i].value,
                                                       dict())
        value = self._convert_value(atom.terms[-1].value)
        parameter_dict[atom.terms[-2].value] = value

    def _convert_value(self, value):
        """
        Converts the value.

        If the value is the form `$<predicate>[<index>]`, returns a function
        to get the size of the type of the `predicate` term defined by `index`.

        If the value is `true` of `false`, returns its boolean form. This
        check is case insensitive.

        Otherwise, return the value.

        :param value: the value
        :type value: str or int or float
        :return: the converted value
        :rtype: int or float or str or bool or function
        """
        if isinstance(value, str):
            match = PREDICATE_TYPE_MATCH.match(value)
            if match is not None:
                groups = match.groups()
                predicate_name = groups[0]
                arity = groups[1]
                index = int(groups[2])
                predicate = Predicate(predicate_name, int(arity))
                if len(groups) > 3 and groups[3] is not None:
                    term = get_constant_from_string(groups[3][1:-1])
                    return self.get_index_of_constant(predicate, index, term)
                else:
                    return self.get_constant_size(predicate, index)

            lower_value = value.lower()
            if lower_value == "true":
                value = True
            elif lower_value == "false":
                value = False

        return value


DEFAULT_PARAMETERS = [
    ("dataset_class", "default_dataset",
     "the class to handle the examples.", False),

    ("avoid_constant", {},
     "Skips adding the constants that appears in the index specified by this "
     "parameter, when reading the examples. "
     "This may cause entities, that only appears in the examples, to not "
     "appear in the knowledge base. "
     "This must be handled by the dataset; otherwise, an exception will be "
     "raised."),

    ("initial_value", {
        "class_name": "random_normal",
        "config": {"mean": 0.5, "stddev": 0.125}
    },
     "initializer for learnable predicates. This initializer will be used to "
     "initialize facts from learnable predicates that are not in the "
     "knowledge base."),

    ("allow_sparse", True,
     "by default, we represent constant facts as sparse matrices whenever "
     "it is possible. Although it reduces the amount of used memory, "
     "it limits the multiplication to matrices with rank at most 2. We case "
     "one needs to work if higher order matrices, this options must be set to "
     "`False`."),

    ("recursion_depth", 1, "the maximum recursion depth for the predicate."),

    ("literal_negation_function", "literal_negation_function",
     "function to get the value of a negated literal from the non-negated "
     "one"),

    ("literal_negation_function:sparse", "literal_negation_function:sparse",
     "function to get the value of a negated literal from the non-negated "
     "one"),

    ("literal_combining_function", "tf.math.add",
     "function to combine the different proves of a literal (FactLayers and "
     "RuleLayers). The default is to sum all the proves, element-wise, by "
     "applying the `tf.math.add` function to reduce the layers outputs"),

    ("and_combining_function", "tf.math.multiply",
     "function to combine different vector and get an `AND` behaviour between "
     "them. The default is to multiply all the paths, element-wise, by "
     "applying the `tf.math.multiply` function"),

    ("path_combining_function", "tf.math.multiply",
     "function to combine different path from a RuleLayer. The default "
     "is to multiply all the paths, element-wise, by applying the "
     "`tf.math.multiply` function"),

    ("edge_neutral_element", {
        "class_name": "tf.constant",
        "config": {"value": 1.0}
    },
     "element used to extract the tensor value of grounded literal "
     "in a rule. The default edge combining function is the element-wise "
     "multiplication. Thus, the neutral element is `1.0`, represented by "
     "`tf.constant(1.0)`", False),

    ("edge_combining_function", "tf.math.multiply",
     "function to extract the value of the fact based on the input. "
     "The default is the element-wise multiplication implemented by the "
     "`tf.math.multiply` function"),

    ("edge_combining_function_2d", "tf.matmul",
     "function to extract the value of the fact based on the input, "
     "for binary facts. The default is the dot multiplication implemented "
     "by the `tf.matmul` function"),
    ("edge_combining_function_2d:sparse", "edge_combining_function_2d:sparse",
     "function to extract the value of the fact based on the input, "
     "for binary facts. The default is the dot multiplication implemented "
     "by the `tf.matmul` function"),

    ("invert_fact_function", "tf.transpose",
     "function to extract the inverse of facts. The default is the "
     "transpose function implemented by `tf.transpose`"),

    ("invert_fact_function:sparse", "tf.sparse.transpose",
     "function to extract the inverse of facts. The default is the "
     "transpose function implemented by `tf.transpose`"),

    ("any_aggregation_function", "any_aggregation_function",
     "function to aggregate the input of the `any` predicate. The default "
     "function is the `tf.reduce_sum`", False),

    ("attributes_combine_function", "tf.math.multiply",
     "function to combine the numeric terms of a fact. "
     "The default function is the `tf.math.multiply`"),

    ("weighted_attribute_combining_function", "tf.math.multiply",
     "function to combine the weights and values of the attribute facts. "
     "The default function is the `tf.math.multiply`"),

    ("output_extract_function", "tf.nn.embedding_lookup",
     "function to extract the value of an atom with a constant at the "
     "last term position. "
     "The default function is the `tf.nn.embedding_lookup`"),

    ("unary_literal_extraction_function", "unary_literal_extraction_function",
     "function to extract the value of unary prediction. "
     "The default is the dot multiplication, implemented by the `tf.matmul`, "
     "applied to the transpose of the literal prediction")
]
