"""
Define the classes to represent the knowledge of the system.
"""

import logging
import re
from collections import OrderedDict
from typing import Dict, Set, List, Tuple, MutableMapping, TypeVar, Any

import numpy as np
from scipy.sparse import csr_matrix

KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.

TRAINABLE_KEY = "$"

WEIGHT_SEPARATOR = "::"
NEGATION_KEY = "not"
IMPLICATION_SIGN = ":-"
END_SIGN = "."
PLACE_HOLDER = re.compile("({[a-zA-Z0-9_-]+})")

logger = logging.getLogger()


def get_term_from_string(string):
    """
    Transforms the string into a term. A variable if it starts with an upper
    case letter, or a constant, otherwise.

    :param string: the string
    :type string: str
    :raise TermMalformedException case the term is malformed
    :return: the term
    :rtype: Term
    """
    if string[0].isupper():
        return Variable(string)
    elif string[0].islower():
        return Constant(string)
    elif string[0] == string[-1] and (string[0] == "'" or string[0] == '"'):
        return Quote(string)
    else:
        raise TermMalformedException()


def build_terms(arguments):
    """
    Builds a list of terms from the arguments.

    :param arguments: the arguments
    :type arguments: list[Term or str]
    :raise TermMalformedException if the argument is neither a term nor a str
    :return: a list of terms
    :rtype: list[Term]
    """
    terms = []
    for argument in arguments:
        if isinstance(argument, Term):
            terms.append(argument)
        elif isinstance(argument, str):
            terms.append(get_term_from_string(argument))
        elif isinstance(argument, float) or isinstance(argument, int):
            terms.append(Number(argument))
        else:
            raise TermMalformedException()

    return terms


class AtomMalformedException(Exception):
    """
    Represents an atom malformed exception.
    """

    def __init__(self, expected, found) -> None:
        """
        Creates an atom malformed exception.

        :param expected: the number of arguments expected
        :type expected: int
        :param found: the number of arguments found
        :type found: int
        """
        super().__init__("Atom malformed, the predicate expects "
                         "{} argument(s) but {} found.".format(expected, found))


class TermMalformedException(Exception):
    """
    Represents an term malformed exception.
    """

    def __init__(self) -> None:
        """
        Creates an term malformed exception.
        """
        super().__init__("Term malformed, the argument must be either a term "
                         "or a string.")


class ClauseMalformedException(Exception):
    """
    Represents an term malformed exception.
    """

    def __init__(self) -> None:
        """
        Creates an term malformed exception.
        """
        super().__init__("Clause malformed, the clause must be an atom, "
                         "a weighted atom or a Horn clause.")


class BadArgumentException(Exception):
    """
    Represents an bad argument exception.
    """

    def __init__(self, value) -> None:
        """
        Creates an term malformed exception.
        """
        super().__init__("Expected a number or a term, got {}.".format(value))


class UnsupportedMatrixRepresentation(Exception):
    """
    Represents an unsupported matrix representation exception.
    """

    def __init__(self, predicate) -> None:
        """
        Creates an unsupported matrix representation exception.
        """
        super().__init__("Unsupported matrix representation for "
                         "predicate: {}.".format(predicate, ))


class Term:
    """
    Represents a logic term.
    """

    def __init__(self, value):
        """
        Creates a logic term.

        :param value: the value of the term
        :type value: str or float
        """
        self.value = value

    def get_name(self):
        """
        Returns the name of the term.
        :return: the name of the term
        :rtype: str
        """
        return self.value

    def is_constant(self):
        """
        Returns true if the term is a constant.

        :return: true if it is a constant, false otherwise
        :rtype: bool
        """
        return False

    def is_template(self):
        """
        Returns true if the term is a template to be instantiated based on
        the knowledge base.
        :return: True if the term is a template, false otherwise
        :rtype: bool
        """
        return False

    def key(self):
        """
        Specifies the keys to be used in the equals and hash functions.
        :return: the keys
        :rtype: Any
        """
        return self.value

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        if isinstance(other, Term):
            return self.key() == other.key()
        return False

    def __str__(self):
        return self.value

    __repr__ = __str__


class Constant(Term):
    """
    Represents a logic constant.
    """

    def __init__(self, value):
        """
        Creates a logic constant.

        :param value: the name of the constant
        :type value: str
        """
        super().__init__(value)

    # noinspection PyMissingOrEmptyDocstring
    def is_constant(self):
        return True

    def __str__(self):
        return self.value

    __repr__ = __str__


class Variable(Term):
    """
    Represents a logic variable.
    """

    def __init__(self, value):
        """
        Creates a logic variable.

        :param value: the name of the variable
        :type value: str
        """
        super().__init__(value)

    # noinspection PyMissingOrEmptyDocstring
    def is_constant(self):
        return False


class Quote(Term):
    """
    Represents a quoted term, that might contain a template.
    """

    def __init__(self, value):
        """
        Creates a quoted term.

        :param value: the value of the term.
        :type value: str
        """
        super().__init__(value[1:-1])
        self._is_template = PLACE_HOLDER.search(value) is not None
        self.quote = value[0]

    # noinspection PyMissingOrEmptyDocstring
    def is_constant(self):
        return not self._is_template

    # noinspection PyMissingOrEmptyDocstring
    def is_template(self):
        return self._is_template

    def __str__(self):
        return self.quote + super().__str__() + self.quote


class Number(Term):
    """
    Represents a number term.
    """

    def __init__(self, value):
        super().__init__(value)

    # noinspection PyMissingOrEmptyDocstring
    def get_name(self):
        return str(self.value)

    # noinspection PyMissingOrEmptyDocstring
    def is_constant(self):
        return True

    # noinspection PyMissingOrEmptyDocstring
    def is_template(self):
        return False

    def __str__(self):
        return str(self.value)


class TemplateTerm(Term):
    """
    Defines a template term, a term with variable parts to be instantiated
    based on the knowledge base.
    """

    def __init__(self, parts):
        """
        Creates a template term.

        :param parts: the parts of the term.
        :type parts: list[str]
        """
        super().__init__("".join(parts))
        self.parts = parts

    # noinspection PyMissingOrEmptyDocstring
    def is_constant(self):
        return False

    # noinspection PyMissingOrEmptyDocstring
    def is_template(self):
        return True


class Predicate:
    """
    Represents a logic predicate.
    """

    def __init__(self, name, arity=0):
        """
        Creates a logic predicate.

        :param name: the name of the predicate
        :type name: str
        :param arity: the arity of the predicate
        :type arity: int
        """
        self.name = name
        self.arity = arity

    def is_template(self):
        """
        Returns true if the predicate is a template to be instantiated based on
        the knowledge base.
        :return: True if the predicate is a template, false otherwise
        :rtype: bool
        """
        return False

    def __str__(self):
        return "{}/{}".format(self.name, self.arity)

    def key(self):
        """
        Specifies the keys to be used in the equals and hash functions.
        :return: the keys
        :rtype: Any
        """
        return self.name, self.arity

    def get_name(self):
        """
        Returns the name of the predicate.

        :return: the name of the predicate
        :rtype: str
        """
        return self.name

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.key() == other.key()
        return False

    __repr__ = __str__


class TemplatePredicate(Predicate):
    """
    Defines a template predicate, a predicate with variable parts to be
    instantiated based on the knowledge base.
    """

    def __init__(self, parts, arity=0):
        """
        Creates a template predicate.

        :param parts: the parts of the predicate.
        :type parts: list[str]
        """
        super().__init__("".join(parts), arity=arity)
        self.parts = parts

    # noinspection PyMissingOrEmptyDocstring
    def is_template(self):
        return True


class Clause:
    """
    Represents a logic clause.
    """

    def is_template(self):
        """
        Returns true if the clause is a template to be instantiated based on
        the knowledge base.
        :return: True if the clause is a template, false otherwise
        :rtype: bool
        """
        return False

    def is_grounded(self):
        """
        Returns true if the clause is grounded. A clause is grounded if all
        its terms are constants.
        :return: true if the clause is grounded, false otherwise
        :rtype: bool
        """
        pass

    def key(self):
        """
        Specifies the keys to be used in the equals and hash functions.
        :return: the keys
        :rtype: Any
        """
        pass

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.key() == other.key()
        return False


class Atom(Clause):
    """
    Represents a logic atom.
    """

    def __init__(self, predicate, *args, weight=1.0) -> None:
        """
        Creates a logic atom.

        :param predicate: the predicate
        :type predicate: str or Predicate
        :param args: the list of terms, if any
        :type args: Term or str
        :param weight: the weight of the atom
        :type weight: float
        :raise AtomMalformedException in case the number of terms differs
        from the arity of the predicate
        """
        if isinstance(predicate, Predicate):
            self.predicate = predicate
            if predicate.arity != len(args):
                raise AtomMalformedException(predicate.arity,
                                             len(args))
        else:
            self.predicate = Predicate(predicate, len(args))

        self.weight = weight
        # noinspection PyTypeChecker
        self.terms = build_terms(args)

    def __getitem__(self, item):
        return self.terms[item]

    # noinspection PyMissingOrEmptyDocstring
    def is_grounded(self):
        for term in self.terms:
            if isinstance(term, Term) and not term.is_constant():
                return False
        return True

    def arity(self):
        """
        Returns the arity of the atom.

        :return: the arity of the atom
        :rtype: int
        """
        return self.predicate.arity

    # noinspection PyMissingOrEmptyDocstring
    def key(self):
        return self.weight, self.predicate, tuple(self.terms)

    def __str__(self):
        if self.terms is None or len(self.terms) == 0:
            if self.weight == 1.0:
                return self.predicate.name
            else:
                return "{}::{}".format(self.weight, self.predicate.name)

        atom = "{}({})".format(self.predicate.name,
                               ", ".join(map(lambda x: str(x), self.terms)))
        if self.weight != 1.0:
            return "{}{}{}".format(self.weight, WEIGHT_SEPARATOR, atom)
        return atom

    # noinspection PyMissingOrEmptyDocstring
    def is_template(self):
        if self.predicate.is_template():
            return True
        if self.terms is None or len(self.terms) == 0:
            return False
        for term in self.terms:
            if term.is_template():
                return True

        return False

    __repr__ = __str__


class Literal(Atom):
    """
    Represents a logic literal.
    """

    def __init__(self, atom, negated=False, trainable=False) -> None:
        """
        Creates a logic literal from an atom.

        :param atom: the atom
        :type atom: Atom
        :param negated: if the literal is negated
        :type negated: bool
        :param trainable: if the literal is to be trained
        :type trainable: bool
        :raise AtomMalformedException in case the number of terms differs
        from the arity of the predicate
        """
        super().__init__(atom.predicate, *atom.terms, weight=atom.weight)
        self.negated = negated
        self.trainable = trainable

    # noinspection PyMissingOrEmptyDocstring
    def key(self):
        # noinspection PyTypeChecker
        return (self.negated, self.trainable) + super().key()

    def __str__(self):
        atom = super().__str__()
        if self.trainable:
            atom = TRAINABLE_KEY + atom
        if self.negated:
            return "{} {}".format(NEGATION_KEY, atom)
        return atom


class AtomClause(Clause):
    """
    Represents an atom clause, an atom that is also a clause. As such,
    it is written with a `END_SIGN` at the end.
    """

    def __init__(self, atom) -> None:
        """
        Creates an atom clause
        :param atom: the atom
        :type atom: Atom
        """
        super().__init__()
        self.atom = atom

    # noinspection PyMissingOrEmptyDocstring
    def is_template(self):
        return self.atom.is_template()

    # noinspection PyMissingOrEmptyDocstring
    def is_grounded(self):
        return self.atom.is_grounded()

    # noinspection PyMissingOrEmptyDocstring
    def key(self):
        return self.atom.key()

    def __str__(self) -> str:
        return self.atom.__str__() + END_SIGN

    def simple_key(self):
        """
        Returns a simplified key of an atom. This key does not include neither
        the weight of the atom nor the value of the numeric terms.

        It is useful to check if an atom was defined multiple times with
        different weights and attribute values.

        :return: the simple key of the atom
        :rtype: tuple[Predicate, tuple[Term]]
        """
        return self.atom.predicate, tuple([None if isinstance(x, Number) else
                                           x for x in self.atom.terms])


class HornClause(Clause):
    """
    Represents a logic horn clause.
    """

    def __init__(self, head, *body) -> None:
        """
        Creates a Horn clause.

        :param head: the head
        :type head: Atom
        :param body: the list of literal, if any
        :type body: Literal
        """
        self.head = head
        self.body = list(body)

    def __getitem__(self, item):
        if item == 0:
            return self.head
        return self.body[item - 1]

    # noinspection PyMissingOrEmptyDocstring
    def key(self):
        key_tuple = self.head.key()
        if self.body is not None and len(self.body) > 0:
            for literal in self.body:
                key_tuple += literal.key()
        return tuple(key_tuple)

    def __str__(self):
        body = Literal(Atom("true")) if self.body is None else self.body

        return "{} {} {}{}".format(self.head, IMPLICATION_SIGN,
                                   ", ".join(map(lambda x: str(x), body)),
                                   END_SIGN)

    # noinspection PyMissingOrEmptyDocstring
    def is_template(self):
        if self.head.is_template():
            return True
        if self.body is None or len(self.body) == 0:
            return False
        for literal in self.body:
            if literal.is_template():
                return True

        return False

    __repr__ = __str__


class TermType:
    """
    Defines the types of a term.
    """

    def __init__(self, variable, number):
        """
        Creates a term type.

        :param variable: true, if the term is variable
        :type variable: bool
        :param number: true, if the term is number
        :type number: bool or None
        """
        self.variable = variable
        self.number = number

    def key(self):
        """
        Specifies the keys to be used in the equals and hash functions.
        :return: the keys
        :rtype: Any
        """
        return self.variable, self.number

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.key() == other.key()
        return False

    def is_compatible(self, other):
        """
        Checks if two types are compatible against each other.
        Two types are compatibles if their number attribute match or one of
        them is none.

        :param other: the other term
        :type other: TermType
        :return: true if they are compatible, false otherwise.
        :rtype: bool
        """
        return self.number is None or other.number is None \
               or self.number == other.number

    def update_type(self, other):
        """
        If `other` is compatible with this type, returns a new type with the
        attributes as an OR of this type and the `other`.

        :param other: the other type
        :type other: TermType
        :return: the update type
        :rtype: TermType
        """
        if not self.is_compatible(other):
            return None
        return TermType(self.variable or other.variable,
                        self.number or other.number)


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


class PredicateTypeError(Exception):
    """
    Represents a predicate type exception.
    """

    def __init__(self, current_atom, previous_atom):
        super().__init__("Found atom whose terms types are incompatible with "
                         "previous types: {} and {}".format(previous_atom,
                                                            current_atom))


class BiDict(dict, MutableMapping[KT, VT]):
    """
    A bi directional dictionary.
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
    Represents a NeuralLog program.
    """

    facts_by_predicate: Dict[Predicate, Dict[Any, Atom]] = dict()
    "The facts"

    clauses: List[HornClause] = []
    "The clauses"

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

    _numeric_representation: Dict[Predicate, csr_matrix] = dict()
    "The matrix representation"

    # Network part
    # TODO: create the neural network representation
    # TODO: update the knowledge part with the weights learned by the neural net

    def __init__(self, clauses):
        """
        Creates a NeuralLog Program.

        :param clauses: the clauses of the program
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
            if isinstance(clause, AtomClause) and clause.is_grounded():
                self._add_predicate(clause.atom)
                fact_dict = self.facts_by_predicate.setdefault(
                    clause.atom.predicate, OrderedDict())
                old_atom = fact_dict.get(clause.simple_key(), None)
                if old_atom is not None:
                    logger.warning("Warning: Atom %s replaced by %s",
                                   old_atom, clause.atom)
                fact_dict[clause.simple_key()] = clause.atom
                self.logic_predicates.add(clause.atom.predicate)
            elif isinstance(clause, HornClause):
                self._add_predicate(clause.head)
                self.logic_predicates.add(clause.head.predicate)
                for atom in clause.body:
                    self._add_predicate(atom)
                    if atom.trainable:
                        self.logic_predicates.add(atom.predicate)
                self.clauses.append(clause)
            else:
                raise ClauseMalformedException()
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
        """
        types = get_predicate_type(atom)
        atom_predicate = atom.predicate
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
        for clause in self.clauses:
            self._get_constants_from_atom(clause.head, _iterable_constants)
            for literal in clause.body:
                self._get_constants_from_atom(literal, _iterable_constants)
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

    def get_matrix_representation(self, predicate):
        """
        Gets the matrix representation for the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the sparse matrix representation of the data for the given
        predicate
        :rtype: csr_matrix or np.matrix or (csr_matrix, csr_matrix)
        """
        return self._numeric_representation \
            .setdefault(predicate, self._build_matrix_representation(predicate))

    def _build_matrix_representation(self, predicate):
        """
        Builds the matrix representation for the predicate

        :param predicate: the predicate
        :type predicate: Predicate
        :raise UnsupportedMatrixRepresentation in the case the predicate is
        not convertible to matrix form
        :return: the matrix representation of the data for the given predicate
        :rtype: csr_matrix or np.matrix or (csr_matrix, csr_matrix)
        """
        arity = self._get_true_arity(predicate)
        if arity == 0:
            return self._propositional_matrix_representation(predicate)
        elif arity == 1:
            if predicate.arity == 1:
                return self._relational_matrix_representation(predicate)
            elif predicate.arity == 2:
                return self._attribute_matrix_representation(predicate)
        elif arity == 2:
            return self._relational_matrix_representation(predicate)
        raise UnsupportedMatrixRepresentation(predicate)

    def _get_true_arity(self, predicate):
        """
        Returns the true arity of the predicate. The number of terms that are
        not number.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the true arity
        :rtype: int
        """
        return sum(1 for i in self.predicates[predicate] if not i.number)

    def _propositional_matrix_representation(self, predicate):
        return np.array(list(map(lambda x: x.weight,
                                 self.facts_by_predicate[predicate].values())))

    def _attribute_matrix_representation(self, predicate):
        """
        Builds the attribute matrix representation for the binary predicate.

        :param predicate: the predicate
        :type predicate: Predicate
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
            weight_data.append(fact.weight)
            attribute_data.append(fact.terms[attribute_index].value)
            entity_indices.append(
                self.iterable_constants.inverse[fact.terms[entity_index]])

        return (
            csr_matrix((weight_data,
                        (entity_indices, [0] * len(weight_data))),
                       shape=(self.number_of_entities, 1)),
            csr_matrix((attribute_data,
                        (entity_indices, [0] * len(attribute_data))),
                       shape=(self.number_of_entities, 1))
        )

    def _relational_matrix_representation(self, predicate):
        """
        Builds the relational matrix representation for the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the matrix representation of the data
        :rtype: csr_matrix
        """
        data = []
        ind = []
        term_range = list(range(predicate.arity))
        for _ in term_range:
            ind.append([])
        for fact in self.facts_by_predicate.get(predicate, dict()).values():
            data.append(fact.weight)
            for i in term_range:
                ind[i].append(self.iterable_constants.inverse[fact.terms[i]])

        if predicate.arity == 1:
            return csr_matrix((data, (ind[0], [0] * len(data))),
                              shape=(self.number_of_entities, 1))

        return csr_matrix((data, tuple(ind)),
                          shape=(self.number_of_entities,) * predicate.arity)
