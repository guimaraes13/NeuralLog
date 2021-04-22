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
Parses the Abstract Syntax Tree.
"""
import logging
from collections import deque

from neurallog.language.language import *

logger = logging.getLogger(__name__)


class BadTermException(KnowledgeException):
    """
    Represents an exception raised by a mal formed term.
    """

    def __init__(self, term, key, substitution):
        """
        Creates a bad term exception.

        :param term: the term
        :type term: str or Term
        :param key: the key
        :type key: str
        :param substitution: the substitution
        :type substitution: str
        """
        super().__init__("Bad term formed when replacing {key} by {sub} on "
                         "{term}.".format(term=term, key="{" + key + "}",
                                          sub=substitution))


class BadClauseException(KnowledgeException):
    """
    Represents an exception raised by a mal formed clause.
    """

    def __init__(self, clause):
        """
        Creates a bad clause exception.

        :param clause: the clause
        """
        super().__init__("Template only supported in Horn clauses. "
                         "Found {}".format(clause.__str__()))


class KeyDict(dict):
    """
    A dictionary to replace fields when formatting a string.
    If it is asked for a key that it does not have, it returns the keys
    surrounded by curly braces, as such, the place on the string remains the
    same.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, k):
        return self.get(k, "{" + k + "}")


def extract_value(string):
    """
    Tries to extract a number value from a string.

    :param string: the string
    :type string: str
    :return: the extracted value, if it success; otherwise, returns the string
    :rtype: str or float
    """
    try:
        value = float(string)
        return Number(value)
    except ValueError:
        return string


def solve_parts(parts, key, substitution):
    """
    Solve the place holders from the parts.

    :param parts: the parts
    :type parts: list[str]
    :param key: the name of the place holder
    :type key: str
    :param substitution: the value of the place holder
    :type substitution: str
    :return: the solved parts
    :rtype: list[str]
    """
    solved_terms = []
    join_terms = ""
    for part in parts:
        if part.startswith("{"):
            if key == part[1:-1]:
                join_terms += substitution
            else:
                if join_terms != "":
                    solved_terms.append(str(join_terms))
                    join_terms = ""
                solved_terms.append(part)
        else:
            join_terms += part
    if join_terms != "":
        solved_terms.append(str(join_terms))

    return solved_terms


def solve_place_holder_term(term, key, substitution):
    """
    Solve the place holder for the term.

    :param term: the term
    :type term: Term
    :param key: the name of the place holder
    :type key: str
    :param substitution: the value of the place holder
    :type substitution: str
    :raise BadTermException: case a bad term is found
    :return: the solved predicate
    :rtype: Term
    """
    if not term.is_template():
        return term
    if isinstance(term, Quote):
        return Quote(term.quote +
                     term.value.format_map(KeyDict({key: substitution})) +
                     term.quote)
    if isinstance(term, TemplateTerm):
        parts = solve_parts(term.parts, key, substitution)
        if len(parts) == 1:
            value = extract_value(parts[0])
            if isinstance(value, str):
                return get_term_from_string(value)
            return value
        else:
            return TemplateTerm(parts)
    if isinstance(term, ListTerms):
        sub_terms = []
        for sub_term in term.items:
            sub_terms.append(
                solve_place_holder_term(sub_term, key, substitution))
        return ListTerms(sub_terms)
    raise BadTermException(term, key, substitution)


def solve_place_holder_predicate(predicate, key, substitution):
    """
    Solve the place holder for the predicate.

    :param predicate: the predicate
    :type predicate: Predicate or TemplatePredicate
    :param key: the name of the place holder
    :type key: str
    :param substitution: the value of the place holder
    :type substitution: str
    :return: the solved predicate
    :rtype: Predicate or TemplatePredicate
    """
    if not predicate.is_template():
        return predicate
    parts = solve_parts(predicate.parts, key, substitution)
    if len(parts) == 1:
        return Predicate(parts[0], predicate.arity)
    else:
        return TemplatePredicate(parts, predicate.arity)


def solve_literal(literal, key, substitution):
    """
    Solves the literal.

    :param literal: the literal
    :type literal: Literal
    :param key: the name of the place holder
    :type key: str
    :param substitution: the value of the place holder
    :type substitution: str
    :return: the solved literal
    :rtype: Literal
    """
    if not literal.is_template():
        return literal
    literal_pred = solve_place_holder_predicate(literal.predicate,
                                                key, substitution)
    literal_terms = []
    for term in literal.terms:
        literal_terms.append(solve_place_holder_term(term, key, substitution))

    return Literal(Atom(literal_pred, *literal_terms), literal.negated)


def solve_place_holder(clause, key, substitution):
    """
    Solve the place_holder specified by `key` in the clause, by replacing
    it by `substitution`.

    :param clause: the clause
    :type clause: HornClause
    :param key: the name of the place_holder
    :type key: str
    :param substitution: the substitution for the place_holder
    :type substitution: str
    :return: The new clause
    :rtype: HornClause
    """
    if clause.head.is_template():
        head_pred = solve_place_holder_predicate(clause.head.predicate,
                                                 key, substitution)
        head_terms = []
        for term in clause.head.terms:
            head_terms.append(solve_place_holder_term(term,
                                                      key, substitution))
        solved_head = Atom(head_pred, *head_terms)
    else:
        solved_head = clause.head
    solved_body = []
    for literal in clause.body:
        solved_body.append(solve_literal(literal, key, substitution))

    return HornClause(solved_head, *solved_body, provenance=clause.provenance)


def solve_place_holders(clause, place_holders):
    """
    Generates a set of clause by replacing the template terms by the
    possible place holders in place_holders.
    :param clause: the clause
    :type clause: HornClause
    :param place_holders: the place holders
    :type place_holders: dict[str, set[str]]
    :return: the set of Horn clauses
    :rtype: set[HornClause]
    """
    queue = deque([clause])
    for key, values in place_holders.items():
        size = len(queue)
        for _ in range(size):
            current = queue.popleft()
            for sub in values:
                solved_clause = solve_place_holder(current, key, sub)
                if solved_clause is not None:
                    queue.append(solved_clause)

    return set(queue)


def ground_placeholders(parts, place_holders_map, *possible_constants,
                        union=False):
    """
    Finds the possible substitution for the place holders found in parts.

    The substitutions are place on the place_holders map.

    :param parts: the parts of the template term
    :type parts: list[str]
    :param place_holders_map: the map containing the place_holders
    :type place_holders_map: dict[str, set[str]]
    :param possible_constants: the possible constants to replace the
    place_holders
    :type possible_constants: str
    :param union: if `True`, the possible substitutions will be the union of
    the existing possibilities with the possibilities found by this method.
    Otherwise, it will be the intersection.
    :type union: bool
    """
    name = ""
    place_holders = []
    for part in parts:
        if part.startswith("{"):
            place_holders.append(part[1:-1])
            name += "(.+)"
        else:
            name += part
    length = len(place_holders)
    name_regex = re.compile(name)
    possible_subs = dict()
    for cons in possible_constants:
        for match in name_regex.finditer(cons):
            for i in range(length):
                possible_subs.setdefault(place_holders[i],
                                         set()).add(match.group(i + 1))

    for k, v in possible_subs.items():
        if k in place_holders_map:
            if union:
                place_holders_map[k] = place_holders_map[k].union(v)
            else:
                place_holders_map[k] = place_holders_map[k].intersection(v)
        else:
            place_holders_map[k] = v
