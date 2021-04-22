#  Copyright 2021 Victor Guimarães
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Handles equivalent clauses.
"""
from collections import Collection
from typing import Optional, Dict, Set, List

from neurallog.language.language import Atom, Term, Literal, HornClause, \
    format_horn_clause
from neurallog.util import OrderedSet
from neurallog.util.clause_utils import will_rule_be_safe, apply_substitution
from neurallog.util.language import does_terms_match, get_unify_map


def get_item_hash(terms, i):
    """
    Computes the hash of the `terms`, based on its index `i`.

    :param terms: the terms
    :type terms: List[Term]
    :param i: the index of the term
    :type i: int
    :return: the hash
    :rtype: int
    """
    if terms[i] is None:
        return 0
    if terms[i].is_constant():
        return hash(terms[i])

    return i + 1


class EquivalentAtom(Atom):
    """
    A container for an atom with improved equals and hash code to detect
    equivalent atoms.
    """

    def __init__(self, atom, fixed_terms):
        """
        Creates an Equivalent Atom.

        :param atom: the atom
        :type atom: Atom
        :param fixed_terms: a set of fixed terms to be treated as constant (
        e.g. consolidated variables from the rule)
        :type fixed_terms: Set[Term]
        """
        super().__init__(atom.predicate, *atom.terms,
                         weight=atom.weight, provenance=atom.provenance)
        self.fixed_terms = fixed_terms
        self._hash_code = self._compute_hash_code()

        self.substitution_map: Optional[Dict[Term, Term]] = None
        "Substitution map created during the invocation of the equals method."

    def _compute_hash_code(self):
        """
        Computes the hash code for the equivalent atom.

        :return: the hash code
        :rtype: int
        """
        hash_code = 1
        hash_code = 31 * hash_code + hash(self.predicate.name)
        if self.terms:
            hash_code = 31 * hash_code + len(self.terms)
            for i in range(len(self.terms)):
                hash_code = 31 * hash_code + get_item_hash(self.terms, i)
        return hash_code

    def __hash__(self):
        return self._hash_code

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        if not isinstance(other, Atom):
            return False

        self.substitution_map = does_terms_match(self, other, self.fixed_terms)
        return self.substitution_map is not None


class EquivalentClauseAtom(EquivalentAtom):
    """
    Clause to extend the concept of equivalent atom, also comparing the
    clause's body
    """

    def __init__(self, atom, body, fixed_terms):
        """
        Creates an Equivalent Clause Atom.

        :param atom: the atom
        :type atom: Atom
        :param body: the body
        :type body: Set[Literal]
        :param fixed_terms: the fixed terms
        :type fixed_terms: Set[Term]
        """
        super().__init__(atom, fixed_terms)
        self.body = body
        self._hash = self._compute_hash()

    def _compute_hash(self):
        _hash = super().__hash__()
        _hash = 31 * _hash * sum(map(lambda x: hash(x), self.body))
        return _hash

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        if not isinstance(other, EquivalentClauseAtom):
            return False
        if not super().__eq__(other):
            return False

        return self.body == other.body


def get_substitution_map(current_atom, equivalent_atom, fixed_terms):
    """
    Gets the substitution map that makes `current_atom` equals to
    `equivalent_atom`.

    As the `current_atom` and the `equivalent_atom` have probably already
    been tested against each other, this method tries to retrieve the sorted
    map that was created during the call of the equals method on the
    `current_atom`.

    :param current_atom: the current atom
    :type current_atom: EquivalentAtom
    :param equivalent_atom: the equivalent atom
    :type equivalent_atom: EquivalentAtom
    :param fixed_terms: the fixed terms
    :type fixed_terms: Set[Term]
    :return: the substitution map
    :rtype: Dict[Term, Term]
    """
    sub_map = current_atom.substitution_map
    if sub_map is not None:
        return sub_map

    return get_unify_map(current_atom, equivalent_atom, fixed_terms)


class EquivalentHornClause:
    """
    A container for a Horn clause with improved information about it
    generation, in order to detect equivalent clauses and make atoms relevant
    to the equivalent clauses also relevant to this one.
    """

    def __init__(self, head, clause_body=None, last_literal=None,
                 substitution_map=None, next_candidate_index=0):
        """
        Creates an Equivalent Horn Clause.

        :param head: the head of the clause
        :type head: Atom
        :param clause_body: the body of the clause
        :type clause_body: Set[Literal] or None
        :param last_literal: the last literal from the body
        :type last_literal: Literal
        :param substitution_map: the substitution maps that originates the
        clause
        :type substitution_map: Dict[Term, Term]
        :param next_candidate_index: the index of the next candidate, in order
        to prevent equivalent clauses by reordering the body
        :type next_candidate_index: int
        """
        self.head: Atom = head
        self.clause_body: Set[Literal] = OrderedSet(clause_body or ())
        self.last_literal: Literal = last_literal

        self.substitution_maps: List[Dict[Term, Term]] = list()
        self.substitution_maps.append(dict(substitution_map or {}))

        self.fixed_terms = set()
        self.fixed_terms.update(self.head.terms)
        for literal in self.clause_body:
            self.fixed_terms.update(literal.terms)

        self.next_candidate_index: int = next_candidate_index
        self.current_candidates: List[Literal] = []
        self.current_substitution_maps: List[Dict[Term, Term]] = []

    # noinspection PyMissingOrEmptyDocstring
    @property
    def horn_clause(self):
        return HornClause(self.head, *self.clause_body)

    def build_initial_clause_candidates(
            self, candidates, skip_atom, skip_clause):
        """
        Creates a list of Horn clauses containing a Horn clause for each
        candidate, skipping equivalent clauses.

        It skips equivalent clauses by checking if the free variables at the
        candidate atom can be renamed to match the free variables of a
        previously atom. If an equivalent atom `A` is detected,
        the substitution map that makes it equals to another previous atom
        `B` is stored along side with `B`. In this case, when a rule from a
        set of candidates is selected for further refinements, it has a
        substitution map that, if applied to the candidates, can make them
        relevant to discarded atoms (like `A`), thus, it can be also
        considered relevant to `B`.

        :param candidates: the list of candidates
        :type candidates: List[Literal]
        :param skip_atom: map to save the previous equivalent atoms
        :type skip_atom: Dict[EquivalentClauseAtom, EquivalentClauseAtom]
        :param skip_clause: map to save the previous equivalent clauses
        :type skip_clause: Dict[EquivalentClauseAtom, EquivalentHornClause]
        :return: a list of clauses, for each candidate
        :rtype: List[EquivalentHornClause]
        """
        horn_clauses = []
        if self.next_candidate_index >= len(candidates):
            return horn_clauses

        for i in range(self.next_candidate_index, len(candidates)):
            if candidates[i].negated:
                continue
            self.find_candidates_by_substitutions(candidates[i], lambda x: True)
            self.processing_substituted_candidates(
                skip_atom, skip_clause, horn_clauses, i)

        return horn_clauses

    def find_candidates_by_substitutions(self, candidate, filter_function):
        """
        Finds the candidates by applying the possible substitution maps. The
        candidates are stored in

        :param candidate: the candidates
        :type candidate: Literal
        :param filter_function: the function to filter the possible literals
        :type filter_function: Callable[[Literal], bool]
        """
        self.current_candidates = []
        self.current_substitution_maps = []
        if self.fixed_terms.isdisjoint(candidate.terms):
            for substitution_map in self.substitution_maps:
                for key, value in substitution_map.items():
                    if key in candidate.terms and value in self.fixed_terms:
                        lit = apply_substitution(candidate, substitution_map)
                        self._test_and_add(
                            lit, filter_function, substitution_map)
        else:
            lit = apply_substitution(candidate, self.substitution_maps[0])
            self._test_and_add(lit, filter_function, self.substitution_maps[0])

    def processing_substituted_candidates(
            self, skip_atom, skip_clause, horn_clauses, candidate_index):
        """
        Process the substituted candidates.

        :param skip_atom: map to save the previous equivalent atoms
        :type skip_atom: Dict[EquivalentClauseAtom, EquivalentClauseAtom]
        :param skip_clause: map to save the previous equivalent clauses
        :type skip_clause: Dict[EquivalentClauseAtom, EquivalentHornClause]
        :param horn_clauses: the list of Horn clauses to append
        :type horn_clauses: List[EquivalentHornClause]
        :param candidate_index: the initial index of the candidate
        :type candidate_index: int
        """
        for j in range(len(self.current_candidates)):
            candidate = self.current_candidates[j]
            if candidate in self.clause_body:
                continue
            current_atom = EquivalentClauseAtom(
                candidate, self.clause_body, self.fixed_terms)
            equivalent_atom = skip_atom.get(current_atom)
            if equivalent_atom is None:
                substitution_map = dict(self.current_substitution_maps[j])

                current_set = OrderedSet(self.clause_body)
                current_set.add(candidate)

                skip_atom[current_atom] = current_atom
                equivalent_horn_clause = EquivalentHornClause(
                    self.head, current_set, candidate,
                    substitution_map, candidate_index + 1)
                skip_clause[current_atom] = equivalent_horn_clause

                horn_clauses.append(equivalent_horn_clause)
            else:
                equivalent_horn_clause = skip_clause.get(equivalent_atom)
                substitution_map = \
                    dict(equivalent_horn_clause.substitution_maps[0])
                sub_map = get_substitution_map(
                    current_atom, equivalent_atom, self.fixed_terms)
                if sub_map is not None:
                    substitution_map.update(sub_map)
                equivalent_horn_clause.substitution_maps.append(
                    substitution_map)

    def _test_and_add(self, literal, filter_function, substitution_map):
        """
        Tests if the literal passes the `filter_function`, if it does,
        adds it to the current candidates and add the `substitution_map` to
        the `self.substitution_maps`.

        :param filter_function: the function to filter the possible literals
        :type filter_function: Callable[[Literal], bool]
        :param literal: the literal
        :type literal: Literal
        :param substitution_map: the substitution map
        :type substitution_map: Dict[Term, Term]
        """
        if filter_function(literal):
            self.current_candidates.append(literal)
            self.current_substitution_maps.append(substitution_map)

    def build_appended_candidates(self, candidates):
        """
        Creates the new candidate rules by adding on possible literal to the
        current rule's body. The current rule is represented by the head and
        body parameters.

        In addition, it skips equivalents sets by checking if the free
        variables at the candidate atom can be renamed to match the free
        variables of a previously selected one. If an equivalent atom `A`
        is detected, the substitution map that makes it equals to another
        previous atom `B` is stored along side with `B`. In this case,
        when a rule from a set of candidates is selected for further
        refinements, it has a substitution map that, if applied to the
        candidates, can make them relevant to discarded atoms (like `A`),
        thus, it can be also considered relevant to `B`.


        :param candidates: the candidates
        :type candidates: Collection[Literal]
        :return: the list of candidate clauses
        :rtype: List[EquivalentHornClause]
        """
        horn_clauses = []
        if not candidates:
            return horn_clauses

        skip_clause = dict()
        skip_atom = dict()
        for candidate in candidates:
            self.find_candidates_by_substitutions(
                candidate,
                lambda x: will_rule_be_safe(self.head, self.clause_body, x))
            self.processing_substituted_candidates(
                skip_atom, skip_clause, horn_clauses, -1)

        return horn_clauses

    def __hash__(self):
        _hash = hash(self.head)
        _hash = 31 * _hash + sum(map(lambda x: hash(x), self.clause_body))
        return _hash

    # noinspection DuplicatedCode
    def __eq__(self, other):
        if id(self) == id(other):
            return True

        if not isinstance(other, EquivalentHornClause):
            return False

        if self.head != other.head:
            return False

        if self.clause_body != other.clause_body:
            return False

        return self.substitution_maps == other.substitution_maps

    def __repr__(self):
        return format_horn_clause(self.head, self.clause_body)
