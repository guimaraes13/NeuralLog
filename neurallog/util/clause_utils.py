"""
Useful methods to deal with Horn clauses.
"""
from collections import Collection, Iterable
from typing import Set, List, Callable, TypeVar

from neurallog.language.language import Literal, Atom, Term, \
    get_term_from_string, HornClause, Number
from neurallog.util.variable_generator import VariableGenerator


def to_variable_atom(atom, variable_generator=None, variable_map=None,
                     not_number_constants_only=False):
    """
    Replace the constant terms of `atom` to variables, mapping equal constants
    to equal variables.

    :param atom: the atom
    :type atom: Atom
    :param variable_generator: the variable generator
    :type variable_generator: Optional[VariableGenerator]
    :param variable_map: the map of constant to variables
    :type variable_map: Optional[Dict[Term, Term]]
    :param not_number_constants_only: replaces only the not numbers constants
    :return: a new atom, with the constant terms replaces by variables
    :rtype: Atom
    """
    if variable_generator is None:
        variable_generator = VariableGenerator()
    if variable_map is None:
        variable_map = dict()
    terms = []
    for term in atom.terms:
        if not_number_constants_only and isinstance(term, Number):
            terms.append(term)
        else:
            variable = variable_map.get(term)
            if variable is None:
                variable = get_term_from_string(next(variable_generator))
            variable_map[term] = variable
            terms.append(variable)

    return Atom(atom.predicate, *terms)


def apply_substitution(literal, substitutions):
    """
    Applies the substitution of terms in the literal and returns a new one.

    :param literal: the literal
    :type literal: Literal
    :param substitutions: the substitution
    :type substitutions: Dict[Term, Term]
    :return: the new literal with the substituted terms
    :rtype: Literal
    """
    terms = []
    for term in literal.terms:
        terms.append(substitutions.get(term, term))

    return Literal(
        Atom(literal.predicate, *terms, weight=literal.weight), literal.negated)


def get_safe_terms(clause_body):
    """
    Return the safe terms of the body of the clause. The safe terms are the
    terms that appears in positive literals in the body of the clause.

    :param clause_body: the body of the clause
    :type clause_body: Iterable[Literal]
    :return: the safe terms
    :rtype: Set[Term]
    """
    terms = set()
    for literal in clause_body:
        if not literal.negated:
            terms.update(literal.terms)

    return terms


def append_non_constant_terms(atom, terms):
    """
    Appends the non-constant terms of `atom` to `terms`.
    :param atom: the atom
    :type atom: Atom
    :param terms: the terms
    :type terms: Set[Term]
    """
    for term in atom.terms:
        if not term.is_constant():
            terms.add(term)


def get_unsafe_terms(head, body):
    """
    Return the unsafe terms of the body of the clause. The unsafe terms are the
    variables that appears in the head of the clause or in negated literals in
    the body of the clause.

    :param head: the head of the clause
    :type head: Atom
    :param body: the body of the clause
    :type body: Iterable[Literal]
    :return: the safe terms
    :rtype: Set[Term]
    """
    terms = set()
    for literal in body:
        if literal.negated:
            append_non_constant_terms(literal, terms)

    append_non_constant_terms(head, terms)

    return terms


def may_rule_be_safe(horn_clause):
    """
    Checks if the `horn_clause` can become safe, by removing literal from its
    body. A Horn clause is safe when all the variables of the clause appear,
    at least once, in a non-negated literal of the body. Including the
    variables in the head of the clause.

    If there are variables on negated literals on the body that do not appear
    on the non-negated ones, the rule can be made safe by removing those
    literals.

    If there is, at least, one variable on the head of the rule that does not
    appears on a non-negated literal on its body, the rule can not become
    safe and this method will return `False`.

    :param horn_clause: the Horn clause
    :type horn_clause: HornClause
    :return: `True` if the clause can become safe; otherwise, `False`
    :rtype: bool
    """
    return get_safe_terms(horn_clause.body).issuperset(horn_clause.head)


def is_rule_safe(horn_clause):
    """
    Checks if the `horn_clause` is safe.

    :param horn_clause: the Horn clause
    :type horn_clause: HornClause
    :return: `True` if it is safe; otherwise, `False`
    :rtype: bool
    """
    return get_safe_terms(horn_clause.body).issuperset(get_unsafe_terms(
        horn_clause.head, horn_clause.body))


def get_non_negated_literals_with_head_variable(horn_clause):
    """
    Gets the literals from the body of the clause that has, at least,
    one variable that appears in the head.

    :param horn_clause: the Horn clause
    :type horn_clause: HornClause
    :return: the literals that has variables that appear in the head
    :rtype: Collection[Literal]
    """
    head_variables = set()
    append_non_constant_terms(horn_clause.head, head_variables)
    literals = set()
    for literal in filter(lambda x: not x.negated, horn_clause.body):
        if not head_variables.isdisjoint(literal.terms):
            literals.add(literal)

    return literals


def will_rule_be_safe(head, body, candidate):
    """
    Checks if a Horn clause will be safe if `candidate` is added to its body.

    :param head: the head of the clause
    :type head: Atom
    :param body: the body of the clause
    :type body: Iterable[Literal]
    :param candidate: the candidate
    :type candidate: Literal
    :return: `True` if the clause is safe with the candidate; otherwise, `False`
    :rtype: bool
    """
    unsafe_terms = get_unsafe_terms(head, body)
    if candidate.negated:
        append_non_constant_terms(candidate, unsafe_terms)

    return get_safe_terms(body).issuperset(unsafe_terms)


E = TypeVar('E')


def find_biggest_gap(elements: List[E], function: Callable[[E], float]):
    """
    Finds the index of the list where the biggest gat between the evaluation
    of the function on the element is.

    :param elements: the elements
    :type elements: List[E]
    :param function: the evaluation function
    :type function: Callable[[E], float]
    :return: the index of the biggest gap, i.e. the number of elements that
    the first part of the list must have to break at the biggest gap
    :rtype: int
    """
    size = len(elements) - 1
    max_index = size
    max_value = 0
    for i in range(size):
        auxiliary = abs(function(elements[i]) - function(elements[i + 1]))
        if auxiliary > max_value:
            max_value = auxiliary
            max_index = i

    return max_index + 1
