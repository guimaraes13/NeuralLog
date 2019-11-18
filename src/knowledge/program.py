"""
Defines a NeuralLog Program.
"""
import collections
import logging
import sys
from collections import OrderedDict, deque
from typing import TypeVar, MutableMapping, Dict, Any, List, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from src.language.language import Number, TermType, Predicate, Atom, \
    HornClause, Term, AtomClause, ClauseMalformedException, TooManyArguments, \
    PredicateTypeError, UnsupportedMatrixRepresentation, Literal

ANY_PREDICATE_NAME = "any"
NO_EXAMPLE_SET = ":none:"

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


def get_disconnected_literals(clause, connected_literals):
    """
    Gets the literals from the `clause` which are disconnected from the
    source and destination variables but are grounded.

    :param clause: the clause
    :type clause: HornClause
    :param connected_literals: the set of connected literals
    :type connected_literals: Set[Literal]
    :return: the list of disconnected literals
    :rtype: List[Literal]
    """
    ground_literals = []
    for literal in clause.body:
        if literal in connected_literals or not literal.is_grounded():
            continue
        ground_literals.append(literal)
    return ground_literals


def build_literal_dictionaries(clause):
    """
    Builds the dictionaries with the literals of the `clause`. This method
    creates two dictionaries, the first one containing the literals that
    connects different terms; and the second one the loop literals,
    which connects the terms to themselves, either by being unary or by having
    equal terms.

    Both dictionaries have the terms in the literals as keys.

    :param clause: the clause
    :type clause: HornClause
    :return: the dictionaries
    :rtype: (Dict[Term, List[Literal]], Dict[Term, List[Literal]])
    """
    binary_literals_by_term = dict()  # type: Dict[Term, List[Literal]]
    loop_literals_by_term = dict()  # type: Dict[Term, List[Literal]]
    for literal in clause.body:
        if literal.arity() == 1:
            dict_to_append = loop_literals_by_term
        elif literal.arity() == 2:
            if literal.terms[0] == literal.terms[-1]:
                dict_to_append = loop_literals_by_term
            else:
                dict_to_append = binary_literals_by_term
        else:
            continue
        for term in set(literal.terms):
            dict_to_append.setdefault(term, []).append(literal)
    return binary_literals_by_term, loop_literals_by_term


def find_paths(partial_paths, destination, binary_literals_by_term,
               completed_paths, visited_literals):
    """
    Finds the paths from `partial_paths` to `destination` by appending the
    literals from `binary_literals_by_term`. The completed paths are stored
    in `completer_paths` while the used literals are stores in
    `visited_literals`.

    Finally, it returns the dead end paths.

    :param partial_paths: The initial partial paths
    :type partial_paths: deque[RulePath]
    :param destination: the destination term
    :type destination: Term
    :param binary_literals_by_term: the literals to be appended
    :type binary_literals_by_term: Dict[Term, List[Literals]]
    :param completed_paths: the completed paths
    :type completed_paths: deque[RulePath]
    :param visited_literals: the visited literals
    :type visited_literals: Set[Literal]
    :return: the dead end paths
    :rtype: List[RulePath]
    """
    dead_end_paths = []  # type: List[RulePath]
    while len(partial_paths) > 0:
        size = len(partial_paths)
        for i in range(size):
            path = partial_paths.popleft()
            path_end = path.path_end()

            if path_end == destination:
                completed_paths.append(path)
                continue

            possible_edges = binary_literals_by_term.get(path_end, [None])
            not_added_path = True
            for literal in possible_edges:
                new_path = path.new_path_with_item(literal)
                if new_path is None:
                    continue
                partial_paths.append(new_path)
                # noinspection PyTypeChecker
                visited_literals.add(literal)
                not_added_path = False

            if not_added_path:
                dead_end_paths.append(path)
    return dead_end_paths


def append_not_in_set(literal, literals, appended):
    """
    Appends `literal` to `literals` if it is not in `appended`.
    Also, updates appended.

    :param literal: the literal to append
    :type literal: Literal
    :param literals: the list to append to
    :type literals: List[Literal]
    :param appended: the set of already appended literals
    :type appended: Set[Literal]
    """
    if literal not in appended:
        literals.append(literal)
        appended.add(literal)


def complete_path_with_any(dead_end_paths, destination, inverted=False):
    """
    Completes the path by appending the special `any` predicate between the
    end of the path and the destination.

    :param dead_end_paths: the paths to be completed
    :type dead_end_paths: collections.Iterable[RulePath]
    :param destination: the destination
    :type destination: Term
    :param inverted: if `True`, append the any predicate with the terms in the
    reversed order
    :type inverted: bool
    :return: the completed paths
    :rtype: List[RulePath]
    """
    completed_paths = []
    for path in dead_end_paths:
        if inverted:
            any_literal = Literal(Atom(ANY_PREDICATE_NAME,
                                       destination, path.path_end()))
        else:
            any_literal = Literal(Atom(ANY_PREDICATE_NAME,
                                       path.path_end(), destination))
        path.append(any_literal)
        completed_paths.append(path)
    return completed_paths


def append_loop_predicates(completed_paths, loop_literals_by_term,
                           visited_literals, reverse_path=False):
    """
    Appends the loop predicates to the path.

    :param completed_paths: the completed paths
    :type completed_paths: deque[RulePath]
    :param loop_literals_by_term: the loop predicates by term
    :type loop_literals_by_term: Dict[Term, List[Literals]]
    :param visited_literals: the set of visited literals to be updated
    :type visited_literals: Set[Literal]
    :param reverse_path: if `True`, reverse the path before processing
    :type reverse_path: bool
    :return: the final paths
    :rtype: List[RulePath]
    """
    final_paths = []  # type: List[RulePath]
    for path in completed_paths:
        if reverse_path:
            path = path.reverse()
        last_reversed = False
        literals = []
        appended = path.literals
        for i in range(len(path.path)):
            last_reversed = path.inverted[i]
            input_term = path[i].terms[-1 if last_reversed else 0]
            output_term = path[i].terms[0 if last_reversed else -1]
            for literal in loop_literals_by_term.get(input_term, []):
                append_not_in_set(literal, literals, appended)
            literals.append(path.path[i])
            for literal in loop_literals_by_term.get(output_term, []):
                append_not_in_set(literal, literals, appended)
                last_reversed = False
            visited_literals.update(appended)
        final_paths.append(RulePath(literals, last_reversed))
    return final_paths


def find_all_forward_paths(source, destination,
                           loop_literals_by_term,
                           binary_literals_by_term, visited_literals):
    """
    Finds all forward paths from `source` to `destination` by using the
    literals in `binary_literals_by_term` and `loop_literals_by_term`.
    If the destination cannot be reached, it includes a special `any`
    predicated to connect the path.

    :param source: the source of the paths
    :type source: Term
    :param destination: the destination of the paths
    :type destination: Term
    :param loop_literals_by_term: the literals that connects different terms
    :type loop_literals_by_term: Dict[Term, List[Literals]]
    :param binary_literals_by_term: the loop literals, which connects the
    terms to themselves
    :type binary_literals_by_term: Dict[Term, List[Literals]]
    :param visited_literals: the set of visited literals
    :type visited_literals: Set[Literal]
    :return: the completed forward paths between source and destination
    :rtype: List[RulePath]
    """
    partial_paths = deque()  # type: deque[RulePath]
    completed_paths = deque()  # type: deque[RulePath]

    initial_path = build_initial_path(source, loop_literals_by_term,
                                      visited_literals)
    for literal in binary_literals_by_term.get(source, []):
        if initial_path is None:
            inverted = literal.terms[-1] == source
            partial_paths.append(RulePath([literal], inverted))
        else:
            partial_paths.append(initial_path.new_path_with_item(literal))
        visited_literals.add(literal)
    if len(partial_paths) == 0 and initial_path is not None:
        partial_paths.append(initial_path)

    dead_end_paths = find_paths(
        partial_paths, destination, binary_literals_by_term,
        completed_paths, visited_literals)

    completed_with_any = complete_path_with_any(
        dead_end_paths, destination, inverted=False)
    for path in completed_with_any:
        completed_paths.append(path)

    return append_loop_predicates(
        completed_paths, loop_literals_by_term, visited_literals,
        reverse_path=False)


def build_initial_path(source, loop_literals_by_term, visited_literals):
    """
    Builds a path with its initial literals, if any.

    :param source: the source of the path
    :type source: Term
    :param loop_literals_by_term: the literals that connects different terms
    :type loop_literals_by_term: Dict[Term, List[Literals]]
    :param visited_literals: the literals added to the path
    :type visited_literals: Set[Literal]
    :return: the path or `None`
    :rtype: RulePath or None
    """
    loop_literals = loop_literals_by_term.get(source, [])
    if len(loop_literals) == 0:
        return None
    initial_path = RulePath([loop_literals[0]])
    visited_literals.add(loop_literals[0])
    for loop_literal in loop_literals[1:]:
        initial_path.append(loop_literal)
        visited_literals.add(loop_literal)
    return initial_path


def find_all_backward_paths(source, destination,
                            loop_literals_by_term,
                            binary_literals_by_term, visited_literals):
    """
    Finds all backward paths from `source` to `destination` by using the
    literals in `binary_literals_by_term` and `loop_literals_by_term`.
    If the destination cannot be reached, it includes a special `any`
    predicated to connect the path.

    :param source: the source of the paths
    :type source: Term
    :param destination: the destination of the paths
    :type destination: Term
    :param loop_literals_by_term: the literals that connects different terms
    :type loop_literals_by_term: Dict[Term, List[Literals]]
    :param binary_literals_by_term: the loop literals, which connects the
    terms to themselves
    :type binary_literals_by_term: Dict[Term, List[Literals]]
    :param visited_literals: the set of visited literals
    :type visited_literals: Set[Literal]
    :return: the completed backward paths between source and destination
    :rtype: List[RulePath]
    """
    partial_paths = deque()  # type: deque[RulePath]
    completed_paths = deque()  # type: deque[RulePath]

    for literal in binary_literals_by_term.get(source, []):
        if literal in visited_literals:
            continue
        inverted = literal.terms[-1] == source
        partial_paths.append(RulePath([literal], inverted))
        visited_literals.add(literal)

    dead_end_paths = find_paths(
        partial_paths, destination, binary_literals_by_term,
        completed_paths, visited_literals)

    for path in complete_path_with_any(
            dead_end_paths, destination, inverted=True):
        completed_paths.append(path)

    return append_loop_predicates(
        completed_paths, loop_literals_by_term, visited_literals,
        reverse_path=True)


def find_clause_paths(clause, inverted=False):
    """
    Finds the paths in the clause.
    :param inverted: if `True`, creates the paths for the inverted rule;
    this is, the rule in the format (output, input). If `False`,
    creates the path for the standard (input, output) rule format.
    :type inverted: bool
    :param clause: the clause
    :type clause: HornClause
    :return: the completed paths between the terms of the clause and the
    remaining grounded literals
    :rtype: (List[RulePath], List[Literal])
    """
    # Defining variables
    source = clause.head.terms[0]
    destination = clause.head.terms[-1]
    if inverted:
        source, destination = destination, source
    binary_literals_by_term, loop_literals_by_term = \
        build_literal_dictionaries(clause)
    visited_literals = set()

    # Finding forward paths
    forward_paths = find_all_forward_paths(
        source, destination, loop_literals_by_term,
        binary_literals_by_term, visited_literals)

    # Finding backward paths
    source, destination = destination, source
    backward_paths = find_all_backward_paths(
        source, destination, loop_literals_by_term,
        binary_literals_by_term, visited_literals)

    ground_literals = get_disconnected_literals(
        clause, visited_literals)

    return forward_paths + backward_paths, ground_literals


class RulePath:
    """
    Represents a rule path.
    """

    path: List[Literal] = list()
    "The path of literals"

    literals: Set[Literal] = set()
    "The set of literals in the path"

    terms: Set[Term]
    "The set of all terms in the path"

    inverted: List[bool]
    """It is True if the correspondent literal is inverted;
    it is false, otherwise"""

    def __init__(self, path, last_inverted=False):
        """
        Initializes a path.

        :param path: the path
        :type path: collections.Iterable[Literal]
        :param last_inverted: if the last literal is inverted
        :type last_inverted: bool
        """
        self.path = list(path)
        self.literals = set(path)
        # self.last_term = self.path[-1].terms[0 if inverted else -1]
        self.terms = set()
        for literal in self.literals:
            self.terms.update(literal.terms)
        self.inverted = self._compute_inverted(last_inverted)

    def _compute_inverted(self, last_inverted):
        inverted = []
        last_term = self.path[-1].terms[0 if last_inverted else -1]
        for i in reversed(range(0, len(self.path))):
            literal = self.path[i]
            if literal.arity() != 1 and literal.terms[0] == last_term:
                inverted.append(True)
                last_term = literal.terms[-1]
            else:
                inverted.append(False)
                last_term = literal.terms[0]

        return list(reversed(inverted))

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

        if item.arity() == 1:
            output_variable = item.terms[0]
            last_inverted = False
        else:
            if item.terms[0] == self.path_end():
                output_variable = item.terms[-1]
                last_inverted = False
            else:
                output_variable = item.terms[0]
                last_inverted = True

        if item.arity() != 1 and output_variable in self.terms:
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
        :rtype: RulePath or None
        """
        path = RulePath(self.path)
        return path if path.append(item) else None

    def path_end(self):
        """
        Gets the term at the end of the path.

        :return: the term at the end of the path
        :rtype: Term
        """
        return self.path[-1].terms[0 if self.inverted[-1] else -1]

    def reverse(self):
        """
        Gets a reverse path.

        :return: the reverse path
        :rtype: RulePath
        """
        not_inverted = self.path[0].arity() != 1 and not self.inverted[0]
        path = RulePath(reversed(self.path), not_inverted)
        return path

    def __getitem__(self, item):
        return self.path.__getitem__(item)

    def __len__(self):
        return self.path.__len__()

    def __str__(self):
        message = ""
        for i in reversed(range(0, len(self.path))):
            literal = self.path[i]
            prefix = literal.predicate.name
            iterator = list(map(lambda x: x.value, literal.terms))
            if self.inverted[i]:
                prefix += "^{-1}"
                iterator = reversed(iterator)

            prefix += "("
            prefix += ", ".join(iterator)
            prefix += ")"
            message = prefix + message
            if i > 0:
                message = ", " + message

        return message

    __repr__ = __str__


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


def get_constants_from_path(clause, iterable_constants):
    """
    Gets all the constant terms from the paths in the clause and adds it
    to `iterable_constants`.

    :param clause: the clause
    :type clause: HornClause
    :param iterable_constants: the iterable constants
    :type iterable_constants: set[Term]
    """
    paths, _ = find_clause_paths(clause)
    for path in paths:
        for term in path.terms:
            if term.is_constant() and not isinstance(term, Number):
                iterable_constants.add(term)


class NeuralLogProgram:
    """
    Represents a NeuralLog language.
    """

    BUILTIN_PREDICATES = {
        "example": [Predicate("example", -1)],
        "learn": [Predicate("learn", 1)],
        "set_parameter": [Predicate("set_parameter", -1)],
        "set_predicate_parameter": [Predicate("set_predicate_parameter", -1)]
    }

    builtin = build_builtin_predicate()

    facts_by_predicate: Dict[Predicate, Dict[Any, Atom]] = dict()
    """The facts. The values of this variable are dictionaries where the key 
    are the predicate and a tuple of the terms and the values are the atoms 
    itself. It was done this way in order to collapse different definitions 
    of the same atom with different weights, in this way, only the last 
    definition will be considered"""

    examples: Dict[str, Dict[Predicate, Dict[Any, Atom]]] = dict()
    """The examples. The values of this variable are dictionaries where the key 
    are the predicate and a tuple of the terms and the values are the atoms 
    itself. It was done this way in order to collapse different definitions 
    of the same atom with different weights, in this way, only the last 
    definition will be considered"""

    clauses_by_predicate: Dict[Predicate, List[HornClause]] = dict()
    "The clauses by predicate"

    constants: Set[Term] = set()
    "All the constants"

    iterable_constants: BiDict[int, Term] = BiDict()
    "The iterable constants"

    predicates: Dict[Predicate, Tuple[TermType]] = dict()
    "All the predicates and their types"

    logic_predicates: Set[Predicate] = set()
    "The logic predicates"

    functional_predicates: Set[Predicate] = set()
    "The functional predicates"

    trainable_predicates: Set[Predicate] = set()
    "The trainable predicates"

    parameters: Dict[Any, Any] = dict()
    "A dictionary with the parameters defined in the program"

    def __init__(self):
        """
        Creates a NeuralLog Program.
        """
        self._last_atom_for_predicate: Dict[Predicate, Atom] = dict()
        # self.add_clauses(clauses)
        # del self._last_atom_for_predicate
        # self.build_program()

    def build_program(self):
        """
        Builds the program after all the clauses had been added.
        """
        self._get_constants()
        self._add_default_parameters()

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
                raise ClauseMalformedException()

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
            if old_atom.context is not None and atom.context is not None:
                logger.warning("Warning: atom %s defined at line %d:%d "
                               "replaced by Atom %s defined at line %d:%d.",
                               old_atom, old_atom.context.start.line,
                               old_atom.context.start.column,
                               atom, atom.context.start.line,
                               atom.context.start.column)
            elif atom.context is None:
                logger.warning("Warning: atom %s defined at line %d:%d "
                               "replaced by Atom %s.",
                               old_atom, old_atom.context.start.line,
                               old_atom.context.start.column, atom)
            elif old_atom.context is None:
                logger.warning("Warning: atom %s replaced by Atom %s defined at"
                               " line %d:%d.", old_atom,
                               atom, atom.context.start.line,
                               atom.context.start.column)
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
        if atom_predicate.arity > 2:
            raise TooManyArguments(atom.context, atom_predicate.arity)
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
        _iterable_constants = set()
        for facts in self.facts_by_predicate.values():
            for fact in facts.values():
                self._get_constants_from_atom(fact, _iterable_constants)
                fact.context = None
        for sets in self.examples.values():
            for examples in sets.values():
                for example in examples.values():
                    self._get_constants_from_atom(example, _iterable_constants,
                                                  is_example=True)
                    example.context = None
        for clauses in self.clauses_by_predicate.values():
            for clause in clauses:
                self._get_constants_from_atom(clause.head, _iterable_constants)
                clause.head.context = None
                for literal in clause.body:
                    self._get_constants_from_atom(literal, _iterable_constants)
                    literal.context = None
                # get_constants_from_path(clause, _iterable_constants)
        self._build_constant_dict(_iterable_constants)

    def _build_constant_dict(self, _iterable_constants):
        """
        Builds the dictionary of iterable constants.

        :param _iterable_constants: the iterable constants
        :type _iterable_constants: collection.Iterable[Term]
        :return: the dictionary of iterable constants
        :rtype: BiDict[int, Term]
        """
        count = 0
        for constant in sorted(_iterable_constants, key=lambda x: x.__str__()):
            self.iterable_constants[count] = constant
            count += 1
        self.number_of_entities = count

    def _get_constants_from_atom(self, atom, iterable_constants,
                                 is_example=False):
        """
        Gets the constants from an atom.

        :param atom: the atom
        :type atom: Atom
        :param iterable_constants: the iterable constants
        :type iterable_constants: set[Term]
        """
        types = self.predicates[atom.predicate]
        for i in range(len(types)):
            if not atom[i].is_constant() or types[i].number:
                continue
            self.constants.add(atom[i])
            if is_example or types[i].variable:
                iterable_constants.add(atom[i])

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
        # entity_index = 1 - attribute_index
        weight_data = []
        attribute_data = []
        entity_indices = []
        for fact in self.facts_by_predicate.get(predicate, dict()).values():
            index = self._check_iterable_terms(fact.terms)
            if index is None:
                continue
            weight_data.append(1.0 if mask else fact.weight)
            attribute_data.append(fact.terms[attribute_index].value)
            entity_indices.append(index[0])

        weights = csr_matrix((weight_data,
                              (entity_indices, [0] * len(weight_data))),
                             shape=(self.number_of_entities, 1),
                             dtype=np.float32)
        if mask:
            return weights

        return (
            weights,
            csr_matrix((attribute_data,
                        (entity_indices, [0] * len(attribute_data))),
                       shape=(self.number_of_entities, 1), dtype=np.float32)
        )

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
            indices = self._check_iterable_terms(fact.terms)
            if indices is None:
                continue
            for i in term_range:
                ind[i].append(indices[i])
            data.append(1.0 if mask else fact.weight)

        if predicate.arity == 1:
            return csr_matrix((data, (ind[0], [0] * len(data))),
                              shape=(self.number_of_entities, 1),
                              dtype=np.float32)

        return csr_matrix((data, tuple(ind)),
                          shape=(self.number_of_entities,) * predicate.arity,
                          dtype=np.float32)

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
            indices = self._check_iterable_terms(fact.terms)
            if indices is None:
                continue
            if indices[0] != indices[1]:
                continue
            ind[0].append(indices[0])
            ind[1].append(0)
            data.append(1.0 if mask else fact.weight)

        return csr_matrix((data, tuple(ind)),
                          shape=(self.number_of_entities, 1), dtype=np.float32)

    def _check_iterable_terms(self, terms):
        """
        Checks if all the terms are iterable constants. If they all are,
        it returns a list of their indices, otherwise, returns `None`.

        :param terms: the terms
        :type terms: list[Term]
        :return: a list with the indices of the terms, if they all are
        iterable constants; otherwise, returns `None`
        :rtype: list[int] or None
        """
        indices = []
        for term in terms:
            if isinstance(term, Number):
                continue
            index = self.iterable_constants.inverse.get(term, None)
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
            index = self.iterable_constants.inverse.get(
                fact.terms[variable_index], None)
            if index is None:
                continue
            ind[0].append(index)
            ind[1].append(0)
            data.append(1.0 if mask else fact.weight)

        return csr_matrix((data, tuple(ind)),
                          shape=(self.number_of_entities, 1), dtype=np.float32)

    def index_for_constant(self, constant):
        """
        Gets the index for the iterable constant.

        :param constant: the iterable constant
        :type constant: Term
        :return: the index of the iterable constant
        :rtype: int
        """
        return self.iterable_constants.inverse[constant]

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
    def _handle_example(self, example, *args, **kwargs):
        """
        Process the builtin `example` predicate.

        :param example: the example clause
        :type example: AtomClause
        """
        atom = Atom(
            example.atom.terms[0].value,
            weight=example.atom.weight,
            *example.atom.terms[1:],
            context=example.atom.context
        )

        example_set = kwargs.get("example_set", NO_EXAMPLE_SET)
        example_dict = self.examples.setdefault(example_set, OrderedDict())
        example_dict = example_dict.setdefault(atom.predicate, OrderedDict())
        key = atom.simple_key()
        old_atom = example_dict.get(key, None)
        if old_atom is not None:
            logger.warning("Warning: example %s defined at line %d:%d "
                           "replaced by Example %s defined at line %d:%d.",
                           old_atom, old_atom.context.start.line,
                           old_atom.context.start.column,
                           atom, atom.context.start.line,
                           atom.context.start.column)
        example_dict[key] = atom

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
        # predicate = get_predicate_from_string(clause.atom.terms[0].get_name())
        atom = clause.atom
        arity = atom.arity()
        if arity < 2:
            return

        parameter_dict = self.parameters
        for i in range(arity - 2):
            parameter_dict = parameter_dict.setdefault(
                atom.terms[i].value, dict())
        value = atom.terms[-1].value
        if value == "True":
            value = True
        elif value == "False":
            value = False
        parameter_dict[atom.terms[-2].value] = value

    # noinspection PyUnusedLocal
    @builtin("set_predicate_parameter")
    def _set_predicate_parameter(self, clause, *args, **kwargs):
        """
        Process the builtin `set_predicate_parameter` predicate.

        :param clause: the set predicate parameter clause
        :type clause: AtomClause
        """
        atom = clause.atom
        arity = atom.arity()
        if arity < 2:
            return

        parameter_dict = self.parameters
        predicate = get_predicate_from_string(atom.terms[0].value)
        parameter_dict = parameter_dict.setdefault(predicate, dict())
        for i in range(1, arity - 2):
            parameter_dict = parameter_dict.setdefault(atom.terms[i].value,
                                                       dict())
        parameter_dict[atom.terms[-2].value] = atom.terms[-1].value

    def _add_default_parameters(self):
        self.parameters.setdefault(
            "initial_value", {
                "class_name": "random_normal",
                "config": {"mean": 0.5, "stddev": 0.125}
            })

        self.parameters.setdefault("literal_negation_function",
                                   "literal_negation_function")
        self.parameters.setdefault("literal_negation_function:sparse",
                                   "literal_negation_function:sparse")

        self.parameters.setdefault("literal_combining_function", "tf.math.add")
        # function to combine the different proves of a literal
        # (FactLayers and RuleLayers). The default is to sum all the
        # proves, element-wise, by applying the `tf.math.add_n` function

        self.parameters.setdefault("and_combining_function",
                                   "tf.math.multiply")
        # function to combine different vector and get an `AND` behaviour
        # between them.
        # The default is to multiply all the paths, element-wise, by applying
        # the `tf.math.multiply` function.

        self.parameters.setdefault("path_combining_function",
                                   "tf.math.multiply")
        # function to combine different path from a RuleLayer. The default
        # is to multiply all the paths, element-wise, by applying the
        # `tf.math.multiply` function

        self.parameters.setdefault("edge_combining_function",
                                   "tf.math.multiply")
        # function to extract the value of the fact based on the input.
        # The default is the element-wise multiplication implemented by the
        # `tf.math.multiply` function

        self.parameters.setdefault("edge_neutral_element",
                                   {
                                       "class_name": "tf.constant",
                                       "config": {"value": 1.0}
                                   })
        # element is used to extract the tensor value of grounded literal
        # in a rule. The default edge combining function is the element-wise
        # multiplication. Thus, the neutral element is `1.0`, represented by
        # `tf.constant(1.0)`.

        self.parameters.setdefault("edge_combining_function_2d", "tf.matmul")
        self.parameters.setdefault("edge_combining_function_2d:sparse",
                                   "edge_combining_function_2d:sparse")
        # function to extract the value of the fact based on the input,
        # for 2d facts. The default is the dot multiplication implemented
        # by the `tf.matmul` function

        # self.parameters.setdefault("attribute_edge_combining_function",
        #                            "tf.math.multiply")
        # function to extract the value of the fact based on the input, for
        # attribute facts.
        # The default is the dot multiplication implemented by the `tf.matmul`
        # function.

        self.parameters.setdefault("invert_fact_function", "tf.transpose")
        self.parameters.setdefault("invert_fact_function:sparse",
                                   "tf.sparse.transpose")
        # function to extract the inverse of a facts. The default is the
        # transpose function implemented by `tf.transpose`

        # self.parameters["any_aggregation_function"] = "tf.reduce_sum"
        self.parameters.setdefault("any_aggregation_function",
                                   "any_aggregation_function")
        # function to aggregate the input of an any predicate. The default
        # function is the `tf.reduce_sum`.

        self.parameters.setdefault("attributes_combine_function",
                                   "tf.math.multiply")
        # function to combine the numeric terms of a fact.
        # The default function is the `tf.math.multiply`.

        self.parameters.setdefault("weighted_attribute_combining_function",
                                   "tf.math.multiply")
        # function to combine the weights and values of the attribute facts.
        # The default function is the `tf.math.multiply`.

        self.parameters.setdefault("output_extract_function",
                                   "tf.nn.embedding_lookup")
        # function to extract the value of an atom with a constant at the
        # last term position.
        # The default function is the `tf.nn.embedding_lookup`.

        self.parameters.setdefault("unary_literal_extraction_function",
                                   "unary_literal_extraction_function")
        # function to extract the value of unary prediction.
        # The default is the dot multiplication, implemented by the `tf.matmul`,
        # applied to the transpose of the literal prediction.
