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
Modifies proposed clauses.
"""
import re
from abc import abstractmethod
from typing import Optional
import hashlib

import \
    neurallog.knowledge.theory.manager.revision.operator.revision_operator as ro
import neurallog.structure_learning.structure_learning_system as sls
from neurallog.knowledge.examples import Examples
from neurallog.language.language import HornClause, Literal, Atom, \
    get_term_from_string, Predicate
from neurallog.util import Initializable


class ClauseModifier(Initializable):
    """
    A clause modifier.
    """

    def __init__(self, learning_system=None):
        """
        Creates a clause modifier.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        """
        self.learning_system = learning_system

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["learning_system"]

    @abstractmethod
    def modify_clause(self, clause, examples):
        """
        Modifies the clause, given the examples.

        :param clause: the clause
        :type clause: HornClause
        :param examples: the examples
        :type examples: Examples
        :return: a new clause
        :rtype: HornClause
        """
        pass

    def __repr__(self):
        return self.__class__.__name__


class ClauseHeadPredicateModifier(ClauseModifier):
    """
    Modifies the predicate of the head of the clause.

    Useful when one wants to indirect learn rules from a set of examples of
    another predicate.

    For instance, supposes one wants to learn a with head
    `a_1(.)`, from a set of examples `a(.)`, to append to a theory that
    contains a rule of the form `a(.) :- a_1(.), ... .`.
    """

    def __init__(self, learning_system=None,
                 old_predicate=None, new_predicate=None):
        """
        Creates a clause modifier.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param old_predicate: the old predicate, a regex to match the term to
        be substituted, if the regex return a group, it will be appended to
        the new predicate ter
        :type old_predicate: str or None
        :param new_predicate: the new predicate
        :type new_predicate: str or None
        """
        super().__init__(learning_system)
        if old_predicate is not None:
            self.old_predicate = old_predicate
        self.new_predicate = new_predicate

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields() + ["old_predicate", "new_predicate"]

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def initialize(self):
        super().initialize()
        self.predicate_regex = re.compile(self.old_predicate)

    # noinspection PyMissingOrEmptyDocstring
    def modify_clause(self, clause, examples):
        head = clause.head
        match = self.predicate_regex.fullmatch(head.predicate.name)
        if match is None:
            return clause
        groups = match.groups()
        if groups:
            new_predicate = groups[0] + self.new_predicate
        else:
            new_predicate = self.new_predicate
        new_head = Atom(new_predicate, *head.terms, weight=head.weight)
        provenance = clause.provenance
        if isinstance(provenance, ro.LearnedClause):
            provenance = provenance.copy()
            provenance.add_modifier(self)
        new_clause = HornClause(new_head, *clause.body, provenance=provenance)

        return new_clause

    def __repr__(self):
        return f"[{super().__repr__()}] {self.old_predicate}" + \
               f" -> {self.old_predicate}{self.new_predicate}"


class AppendLiteralModifier(ClauseModifier):
    """
    Appends a literal to the end of the clause.

    Useful to append a function literal to the clause, to pose as an
    activation function. The `term_index` field allows one to specify a term,
    of the head of the clause, to be in the appended literal.
    """

    OPTIONAL_FIELDS = dict(ClauseModifier.OPTIONAL_FIELDS)
    OPTIONAL_FIELDS.update({
        "append_at_beginning": False
    })

    def __init__(self, learning_system=None, predicate=None,
                 term_index=None, append_at_beginning=None):
        """
        Creates a clause modifier.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param predicate: the predicate of the literal to append
        :type predicate: str or None
        :param term_index: the index of a term in the head of the clause,
        to be used as the term of the appended literal
        :type term_index: Optional[int]
        :param append_at_beginning: if `True`, appends the literal at the
        beginning of the clause's body
        :type append_at_beginning: Optional[bool]
        """
        super().__init__(learning_system)
        self.predicate = predicate
        self.term_index = term_index

        self.append_at_beginning = append_at_beginning
        if self.append_at_beginning is None:
            self.append_at_beginning = \
                self.OPTIONAL_FIELDS["append_at_beginning"]

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields() + ["predicate"]

    # noinspection PyMissingOrEmptyDocstring
    def modify_clause(self, clause, examples):
        if self.term_index is None:
            literal = Literal(Atom(self.predicate))
        else:
            literal = Literal(
                Atom(self.predicate, clause.head.terms[self.term_index]))
        if literal not in clause.body:
            if self.append_at_beginning:
                clause.body.insert(0, literal)
            else:
                clause.body.append(literal)
            if isinstance(clause.provenance, ro.LearnedClause):
                clause.provenance.add_modifier(self)
        return clause

    def __repr__(self):
        return f"[{super().__repr__()}] {self.predicate}" + \
               f"/1[{self.term_index}]" if self.term_index is not None else ""


class AppendLiteralWithUniqueTermModifier(ClauseModifier):
    """
    Appends a literal to the end of the clause, with a unique generated term.

    The term is only guaranteed to be unique inside an single execution of this
    class, this is, this class won`t generated the same term more than once
    at the same execution, but the term might already exist in the theory.
    """

    OPTIONAL_FIELDS = dict(ClauseModifier.OPTIONAL_FIELDS)
    OPTIONAL_FIELDS.update({
        "_counter": 0,
        "term_prefix": "w_",
        "append_at_beginning": False
    })

    def __init__(self, learning_system=None, predicate=None,
                 term_prefix=None, append_at_beginning=None):
        """
        Creates a clause modifier.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param predicate: the predicate of the literal to append
        :type predicate: str or None
        :param term_prefix: the prefix of the term
        :type term_prefix: Optional[str]
        :param append_at_beginning: if `True`, appends the literal at the
        beginning of the clause's body
        :type append_at_beginning: Optional[bool]
        """
        super().__init__(learning_system)
        self._counter = self.OPTIONAL_FIELDS["_counter"]
        self.predicate = predicate
        self.literal_predicate: Optional[Predicate] = None
        self.term_prefix = term_prefix
        if self.term_prefix is None:
            self.term_prefix = self.OPTIONAL_FIELDS["term_prefix"]

        self.append_at_beginning = append_at_beginning
        if self.append_at_beginning is None:
            self.append_at_beginning = \
                self.OPTIONAL_FIELDS["append_at_beginning"]

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields() + ["predicate"]

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def initialize(self):
        super().initialize()
        self.literal_predicate = Predicate(self.predicate, 1)

    def _contains_predicate(self, clause):
        """
        Checks if the clause contains the predicate to be added.

        :param clause: the clause
        :type clause: HornClause
        :return: `True`, if the clause contains the predicate
        :rtype: bool
        """
        for literal in clause.body:
            if self.literal_predicate == literal.predicate:
                return True

        return False

    # noinspection PyMissingOrEmptyDocstring
    def modify_clause(self, clause, examples):
        if not self._contains_predicate(clause):
            literal = Literal(Atom(self.predicate,
                                   get_term_from_string(
                                       self.term_prefix + str(self._counter))))
            if self.append_at_beginning:
                clause.body.insert(0, literal)
            else:
                clause.body.append(literal)
            if isinstance(clause.provenance, ro.LearnedClause):
                clause.provenance.add_modifier(self)
            self._counter += 1
        return clause

    def __repr__(self):
        return f"[{super().__repr__()}] {self.predicate}/1[" + \
               f"{self.term_prefix} + ()]"


class AppendHashLiteralTermModifier(ClauseModifier):
    """
    Appends a literal term, whose name is computed from the hash of the clause,
    to the end of the clause, with term that can be either a constant or
    extracted from the clause.
    """

    OPTIONAL_FIELDS = dict(ClauseModifier.OPTIONAL_FIELDS)
    OPTIONAL_FIELDS.update({
        "literal_prefix": "w_",
        "head_term_index": None,
        "hash_length": 4,
    })

    def __init__(self, learning_system=None,
                 literal_prefix=None, head_term_index=None, hash_length=None):
        """
        Creates a clause modifier.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param literal_prefix: the prefix of the literal
        :type literal_prefix: Optional[str]
        :param head_term_index: if not `None`, adds the term of index from
        the head of the clause to the created predicate.
        :type head_term_index: Optional[int]
        :param hash_length: Sets the length of the hash to be appended to the
        literal name
        :type hash_length: Optional[int]
        """
        super().__init__(learning_system)
        self.literal_prefix = literal_prefix
        if self.literal_prefix is None:
            self.literal_prefix = self.OPTIONAL_FIELDS["literal_prefix"]

        self.head_term_index = head_term_index
        if self.append_at_beginning is None:
            self.append_at_beginning = self.OPTIONAL_FIELDS["head_term_index"]

        self.hash_length = hash_length
        if self.hash_length is None:
            self.hash_length = self.OPTIONAL_FIELDS["hash_length"]

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields()

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def initialize(self):
        super().initialize()

    def _contains_predicate(self, clause):
        """
        Checks if the clause contains a predicate with the same prefix.

        :param clause: the clause
        :type clause: HornClause
        :return: `None`, if the clause does not contain the predicate;
        otherwise, the literal of the predicate
        :rtype: Optional[Literal]
        """
        for literal in clause.body:
            if literal.predicate.name.startswith(self.literal_prefix):
                return literal

        return None

    # noinspection PyMissingOrEmptyDocstring
    def modify_clause(self, clause, examples):
        old_literal = self._contains_predicate(clause)
        if old_literal is not None:
            clause.body.remove(old_literal)

        new_literal_name = hashlib.sha1(str(clause).encode("utf-8")).hexdigest()
        new_literal_name = new_literal_name[:self.hash_length]

        terms = []
        if self.head_term_index is not None:
            terms.append(clause.head.terms[self.head_term_index])

        literal = Literal(Atom(self.literal_prefix + new_literal_name, *terms))
        clause.body.append(literal)
        if isinstance(clause.provenance, ro.LearnedClause):
            clause.provenance.add_modifier(self)

        return clause

    def __repr__(self):
        if self.head_term_index is None:
            return f"[{super().__repr__()}] {self.literal_prefix}"
        else:
            return f"[{super().__repr__()}] {self.literal_prefix}[" + \
                   f"{self.head_term_index}]"
