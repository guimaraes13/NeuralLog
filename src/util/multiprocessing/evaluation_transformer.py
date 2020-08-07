"""
Handles the transformation of the evaluation object into async theory
evaluator.
"""
import collections
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Set

from src.knowledge.examples import Examples
from src.knowledge.theory.manager.revision.clause_modifier import ClauseModifier
from src.language.equivalent_clauses import EquivalentHornClause
from src.language.language import Literal, HornClause
from src.util.multiprocessing.theory_evaluation import AsyncTheoryEvaluator, \
    SyncTheoryEvaluator

V = TypeVar('V')
E = TypeVar('E')
J = TypeVar('J')
K = TypeVar('K')


def apply_modifiers(modifiers, clause, examples):
    """
    Applies the clause modifiers to the clause.

    :param modifiers: the clause modifiers
    :type modifiers: collections.Iterable[ClauseModifier]
    :param clause: the clause
    :type clause: HornClause
    :param examples: the examples
    :type examples: Examples
    :return: the modified clause
    :rtype: HornClause
    """
    for clause_modifier in modifiers:
        clause = clause_modifier.modify_clause(clause, examples)
    return clause


class AsyncEvaluationTransformer(ABC, Generic[V, E]):
    """
    Class to encapsulate instances of type `TV` into `AsyncTheoryEvaluator`s.
    """

    def __init__(self, clause_modifiers=None):
        """
        Creates an async evaluation transformer.

        :param clause_modifiers: the clause modifiers
        :type clause_modifiers: ClauseModifier or Collection[ClauseModifier] or
        None
        """
        self._clause_modifiers = None
        self.clause_modifiers = clause_modifiers

    @abstractmethod
    def transform(self, evaluator, v, examples):
        """
        Transforms a element of type `V` into an `AsyncTheoryEvaluator`.

        :param evaluator: the evaluator
        :type evaluator: SyncTheoryEvaluator[E] or AsyncTheoryEvaluator[E]
        :param v: the element
        :type v: V
        :param examples: the examples
        :type examples: Examples
        :return: the async theory evaluator
        :rtype: SyncTheoryEvaluator[E] or AsyncTheoryEvaluator[E]
        """
        pass

    @property
    def clause_modifiers(self):
        """
        Gets the clause modifiers.

        :return: the clause modifiers
        :rtype: List[ClauseModifier]
        """
        return self._clause_modifiers

    @clause_modifiers.setter
    def clause_modifiers(self, value):
        """
        Sets the clause modifiers.

        :param value: the clause modifiers
        :type value: Optional[ClauseModifier or List[ClauseModifier]]
        """
        if not value:
            self._clause_modifiers = []
        else:
            if isinstance(value, ClauseModifier):
                self._clause_modifiers = [value]
            else:
                self._clause_modifiers = list(value)


class EquivalentHonClauseAsyncTransformer(AsyncEvaluationTransformer[
                                              EquivalentHornClause,
                                              EquivalentHornClause]):
    """
    Transforms a `EquivalentHornClause` into an
    `AsyncTheoryEvaluator[EquivalentHornClause]`
    """

    # noinspection PyMissingOrEmptyDocstring
    def transform(self, evaluator: AsyncTheoryEvaluator[EquivalentHornClause],
                  equivalent_horn_clause: EquivalentHornClause,
                  examples: Examples):
        """
        Transforms an equivalent Horn clause into an `AsyncTheoryEvaluator`.

        :param evaluator: the evaluator
        :type evaluator: SyncTheoryEvaluator[E] or
        AsyncTheoryEvaluator[E]
        :param equivalent_horn_clause: the equivalent Horn clause
        :type equivalent_horn_clause: EquivalentHornClause
        :param examples: the examples
        :type examples: Examples
        :return: the async theory evaluator
        :rtype: SyncTheoryEvaluator[E] or AsyncTheoryEvaluator[E]
        """
        evaluator.horn_clause = apply_modifiers(
            self.clause_modifiers, equivalent_horn_clause.horn_clause, examples)
        evaluator.element = equivalent_horn_clause

        return evaluator


class LiteralAppendAsyncTransformer(AsyncEvaluationTransformer[Literal, J]):
    """
    Encapsulates extended HornClauses, from a initial Horn clause and a new
    literal, into AsyncTheoryEvaluators.
    """

    def __init__(self, initial_clause=None, clause_modifiers=None):
        """
        Creates a Literal Append Async Transformer.

        :param initial_clause: the initial clause
        :type initial_clause: HornClause or None
        :param clause_modifiers: the clause modifiers
        :type clause_modifiers: ClauseModifier or Collection[ClauseModifier] or
        None
        """
        super().__init__(clause_modifiers)
        self.initial_clause: Optional[HornClause] = initial_clause

    # noinspection PyMissingOrEmptyDocstring
    def transform(self, evaluator, literal, examples):
        """
        Transforms a literal element into an `AsyncTheoryEvaluator`.

        :param evaluator: the evaluator
        :type evaluator: SyncTheoryEvaluator[E] or
        AsyncTheoryEvaluator[E]
        :param literal: the literal
        :type literal: Literal
        :param examples: the examples
        :type examples: Examples
        :return: the async theory evaluator
        :rtype: SyncTheoryEvaluator[E] or AsyncTheoryEvaluator[E]
        """
        clause = HornClause(
            self.initial_clause.head, *list(self.initial_clause.body))
        clause.body.append(literal)
        clause = apply_modifiers(self.clause_modifiers, clause, examples)
        evaluator.horn_clause = clause

        return evaluator


class ConjunctionAppendAsyncTransformer(AsyncEvaluationTransformer[
                                            Set[Literal], K]):
    """
    Encapsulates extended Horn clauses, from an initial Horn clause and a new
    literal, into AsyncTheoryEvaluators.
    """

    def __init__(self, initial_clause=None, clause_modifiers=None):
        """
        Creates a Literal Append Async Transformer.

        :param initial_clause: the initial clause
        :type initial_clause: HornClause or None
        :param clause_modifiers: the clause modifiers
        :type clause_modifiers: ClauseModifier or Collection[ClauseModifier] or
        None
        """
        super().__init__(clause_modifiers)
        self.initial_clause: Optional[HornClause] = initial_clause

    def transform(self, evaluator, conjunction, examples):
        """
        Transforms a conjunction of literals into an `AsyncTheoryEvaluator`.

        :param evaluator: the evaluator
        :type evaluator: SyncTheoryEvaluator[E] or
        AsyncTheoryEvaluator[E]
        :param conjunction: the conjunction of literals
        :type conjunction: Set[Literal]
        :param examples: the examples
        :type examples: Examples
        :return: the async theory evaluator
        :rtype: SyncTheoryEvaluator[E] or AsyncTheoryEvaluator[E]
        """
        clause = HornClause(
            self.initial_clause.head, *list(self.initial_clause.body))
        for literal in conjunction:
            if literal not in clause.body:
                clause.body.append(literal)
        clause = apply_modifiers(self.clause_modifiers, clause, examples)
        evaluator.horn_clause = clause

        return evaluator
