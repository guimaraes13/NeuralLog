"""
Useful method to deal with the logic language.
"""
from typing import Set, Dict

from src.language.language import Atom, Term


def get_unify_map(atom, goal, fixed_terms=None):
    """
    Tries to unify the given atom to the given goal. If it is possible, returns
    the substitution map of the terms that unifies them; otherwise, returns
    `None`.

    This method is not symmetric,
    i.e. `get_unify_map(a, b, c) != get_unify_map(b, a, c)`.

    :param atom: the atom
    :type atom: Atom
    :param goal: the goal
    :type goal: Atom
    :param fixed_terms: a set of terms to be treated as constant
    :type fixed_terms: Optional[Set[Term]]
    :return: the substitution map that makes `atom` equals to `goal`,
    given the fixed terms, if it exists
    :rtype: Optional[Dict[Term, Term]]
    """
    variable_map: Dict[Term, Term] = dict()
    fixed_terms = fixed_terms or set()

    for goal_term, atom_term in zip(goal.terms, atom.terms):
        if goal_term.is_constant() or goal_term in fixed_terms:
            # The goal's term is a constant, the atom term must exactly match it
            if goal_term != atom_term:
                return None
        else:
            # The goal's term is a variable
            mapped_term = variable_map.get(atom_term, None)
            if mapped_term is not None:
                # The atom's term has been already mapped to another term
                # The mapped term must exactly match the goal's term
                if goal_term != mapped_term:
                    return None
            else:
                # The atom's term has not yet been mapped
                # It is then mapped to the goal's variable
                variable_map[atom_term] = goal_term

    return variable_map


def does_terms_match(goal, atom, fixed_terms):
    """
    Check if the constants and variables from the goal match the ones from
    the atom,

    :param goal: the goal
    :type goal: Atom
    :param atom: the atom
    :type atom: Atom
    :param fixed_terms: the fixed terms
    :type fixed_terms: Set[Term]
    :return: a map to make the goal equals to the atom, if the goal matches;
    otherwise, returns `None`
    :rtype: Optional[Dict[Term, Term]]
    """
    if goal.predicate != atom.predicate or \
            get_unify_map(atom, goal, fixed_terms) is None:
        return None
    return get_unify_map(goal, atom, fixed_terms)


def iterable_to_string(iterable, separator="\n"):
    """
    Transforms the `iterable` to a string by getting the string value of
    each item and joining them with `sep`.

    :param iterable: the iterable
    :type iterable: Any
    :param separator: the separator
    :type separator: str
    :return: the joined string
    :rtype: str
    """
    return separator.join(map(lambda x: str(x), iterable))
