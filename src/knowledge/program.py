"""
Defines a NeuralLog Program.
"""
import logging
from collections import OrderedDict

import numpy as np
from scipy.sparse import csr_matrix
from typing import TypeVar, MutableMapping, Dict, Any, List, Set, Tuple

from src.language.language import Number, TermType, Predicate, Atom, \
    HornClause, Term, AtomClause, ClauseMalformedException, TooManyArguments, \
    PredicateTypeError, UnsupportedMatrixRepresentation

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


# noinspection DuplicatedCode
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
        "set_predicate_parameter": [Predicate("set_predicate_parameter", -1)]
    }

    builtin = build_builtin_predicate()

    facts_by_predicate: Dict[Predicate, Dict[Any, Atom]] = dict()
    """The facts. The values of this variable are dictionaries where the key 
    are the predicate and a tuple of the terms and the values are the atoms 
    itself. It was done this way in order to collapse different definitions 
    of the same atom with different weights, in this way, only the last 
    definition will be considered"""

    examples: Dict[Predicate, Dict[Any, Atom]] = dict()
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

    parameters: Dict[Any, Any] = {
        "initial_value": {
            "class_name": "random_normal",
            "config": {"mean": 0.5, "stddev": 0.125}
        }
    }
    "A dictionary with the parameters defined in the program"

    def __init__(self, clauses):
        """
        Creates a NeuralLog Program.

        :param clauses: the clauses of the language
        :type clauses: collections.iterable[Clause]
        """
        self._last_atom_for_predicate: Dict[Predicate, Atom] = dict()
        self._process_clauses(clauses)
        del self._last_atom_for_predicate
        self._get_constants()

    def _process_clauses(self, clauses):
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
                    self._process_builtin_clause(clause)
                    continue

            if isinstance(clause, AtomClause) and clause.is_grounded():
                self._add_predicate(clause.atom)
                fact_dict = self.facts_by_predicate.setdefault(
                    clause.atom.predicate, OrderedDict())
                old_atom = fact_dict.get(clause.simple_key(), None)
                if old_atom is not None:
                    logger.warning("Warning: atom %s defined at line %d:%d "
                                   "replaced by Atom %s defined at line %d:%d.",
                                   old_atom, old_atom.context.start.line,
                                   old_atom.context.start.column,
                                   clause.atom, clause.atom.context.start.line,
                                   clause.atom.context.start.column)
                fact_dict[clause.simple_key()] = clause.atom
                self.logic_predicates.add(clause.atom.predicate)
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
        for examples in self.examples.values():
            for example in examples.values():
                self._get_constants_from_atom(example, _iterable_constants)
                example.context = None
        for clauses in self.clauses_by_predicate.values():
            for clause in clauses:
                self._get_constants_from_atom(clause.head, _iterable_constants)
                clause.head.context = None
                for literal in clause.body:
                    self._get_constants_from_atom(literal, _iterable_constants)
                    literal.context = None
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

    def _get_constants_from_atom(self, atom, iterable_constants):
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
            if types[i].variable:
                iterable_constants.add(atom[i])

    # TODO: get the matrix representation for the literal instead of the
    #  predicate:
    #  - adjust for negated literals;
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
            index = self._check_iterable_terms(fact.terms)
            if index is None:
                continue
            weight_data.append(1.0 if mask else fact.weight)
            attribute_data.append(fact.terms[attribute_index].value)
            entity_indices.append(index[0])

        weights = csr_matrix((weight_data,
                              (entity_indices, [0] * len(weight_data))),
                             shape=(self.number_of_entities, 1))
        if mask:
            return weights

        return (
            weights,
            csr_matrix((attribute_data,
                        (entity_indices, [0] * len(attribute_data))),
                       shape=(self.number_of_entities, 1))
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
                              shape=(self.number_of_entities, 1))

        return csr_matrix((data, tuple(ind)),
                          shape=(self.number_of_entities,) * predicate.arity)

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
                          shape=(self.number_of_entities, 1))

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

    def get_vector_representation_with_constant(self, atom):
        """
        Gets the vector representation for a binary atom with a constant and
        a variable.

        :param atom: the binary atom
        :type atom: Atom
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
            data.append(fact.weight)

        return csr_matrix((data, tuple(ind)),
                          shape=(self.number_of_entities, 1))

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

    def _process_builtin_clause(self, clause):
        """
        Process the builtin clause.

        :param clause: the clause
        :type clause: AtomClause
        """
        predicate = clause.atom.predicate
        # noinspection PyUnresolvedReferences
        self.builtin.functions[predicate.name](self, clause)

    @builtin("example")
    def _handle_example(self, example):
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
        example_dict = self.examples.setdefault(atom.predicate, OrderedDict())
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

    @builtin("learn")
    def _learn_predicate(self, clause):
        """
        Process the builtin `learn` predicate.

        :param clause: the learn clause
        :type clause: AtomClause
        """
        predicate = get_predicate_from_string(clause.atom.terms[0].get_name())
        self.trainable_predicates.add(predicate)

    @builtin("set_parameter")
    def _set_parameter(self, clause):
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
        parameter_dict[atom.terms[-2].value] = atom.terms[-1].value

    @builtin("set_predicate_parameter")
    def _set_predicate_parameter(self, clause):
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
        parameter_dict = parameter_dict.get(predicate, dict())
        for i in range(1, arity - 2):
            parameter_dict = parameter_dict.get(atom.terms[i].value, dict())
        parameter_dict[atom.terms[-2].value] = atom.terms[-1].value
