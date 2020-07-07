"""
Modifies proposed clauses.
"""
import re
from abc import abstractmethod

import src.knowledge.theory.manager.revision.operator.revision_operator as ro
import src.structure_learning.structure_learning_system as sls
from src.knowledge.examples import Examples
from src.language.language import HornClause, Literal, Atom, \
    get_term_from_string
from src.util import Initializable


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
        :type old_predicate: Optional[str]
        :param new_predicate: the new predicate
        :type new_predicate: Optional[int]
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
        return f"[{super().__repr__()}] {self.new_predicate}" + \
               f" -> () + {self.old_predicate}"


class AppendLiteralModifier(ClauseModifier):
    """
    Appends a literal to the end of the clause.

    Useful to append a function literal to the clause, to pose as an
    activation function. The `term_index` field allows one to specify a term,
    of the head of the clause, to be in the appended literal.
    """

    def __init__(self, learning_system=None, predicate=None, term_index=None):
        """
        Creates a clause modifier.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param predicate: the predicate of the literal to append
        :type predicate: Optional[str]
        :param term_index: the index of a term in the head of the clause,
        to be used as the term of the appended literal
        :type term_index: Optional[int]
        """
        super().__init__(learning_system)
        self.predicate = predicate
        self.term_index = term_index

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

    OPTIONAL_FIELDS = ClauseModifier.OPTIONAL_FIELDS
    OPTIONAL_FIELDS.update({
        "_counter": 0,
        "term_prefix": "w_"
    })

    def __init__(self, learning_system=None, predicate=None, term_prefix=None):
        """
        Creates a clause modifier.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param predicate: the predicate of the literal to append
        :type predicate: Optional[str]
        :param term_prefix: the prefix of the term
        :type term_prefix: Optional[str]
        """
        super().__init__(learning_system)
        self._counter = self.OPTIONAL_FIELDS["_counter"]
        self.predicate = predicate
        self.term_prefix = term_prefix
        if self.term_prefix is None:
            self.term_prefix = self.OPTIONAL_FIELDS["term_prefix"]

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields() + ["predicate"]

    # noinspection PyMissingOrEmptyDocstring
    def modify_clause(self, clause, examples):
        literal = Literal(
            Atom(self.predicate,
                 get_term_from_string(self.term_prefix + str(self._counter))))
        if literal.predicate not in map(lambda x: x.predicate, clause.body):
            clause.body.append(literal)
            if isinstance(clause.provenance, ro.LearnedClause):
                clause.provenance.add_modifier(self)
        return clause

    def __repr__(self):
        return f"[{super().__repr__()}] {self.predicate}/1[" + \
               f"{self.term_prefix} + ()]"
