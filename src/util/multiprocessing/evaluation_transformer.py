"""
Handles the transformation of the evaluation object into async theory
evaluator.
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional

from src.knowledge.examples import Examples
from src.knowledge.theory.manager.revision.clause_modifier import ClauseModifier
from src.language.equivalent_clauses import EquivalentHornClause
from src.language.language import Literal, HornClause
from src.util.multiprocessing.theory_evaluation import AsyncTheoryEvaluator, \
    SyncTheoryEvaluator

V = TypeVar('V')
E = TypeVar('E')


class AsyncEvaluationTransformer(ABC, Generic[V, E]):
    """
    Class to encapsulate instances of type `TV` into `AsyncTheoryEvaluator`s.
    """

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


class EquivalentHonClauseAsyncTransformer(AsyncEvaluationTransformer[
                                              EquivalentHornClause,
                                              EquivalentHornClause]):
    """
    Transforms a `EquivalentHornClause` into an
    `AsyncTheoryEvaluator[EquivalentHornClause]`
    """

    # noinspection PyMissingOrEmptyDocstring
    def transform(self, evaluator: AsyncTheoryEvaluator[EquivalentHornClause],
                  v: EquivalentHornClause, examples: Examples):
        evaluator.horn_clause = v.horn_clause
        evaluator.element = v

        return evaluator


K = TypeVar('K')


class LiteralAppendAsyncTransformer(AsyncEvaluationTransformer[Literal, K]):
    """
    Encapsulates extended HornClauses, from a initial Horn clause and a new
    literal, into AsyncTheoryEvaluators.
    """

    def __init__(self, clause_modifiers=None):
        """
        Creates a Literal Append Async Transformer.

        :param clause_modifiers: the clause modifiers
        :type clause_modifiers: Optional[ClauseModifier or List[ClauseModifier]]
        """
        self.initial_clause: Optional[HornClause] = None
        self._clause_modifiers = None
        self.clause_modifiers = clause_modifiers

    # noinspection PyMissingOrEmptyDocstring
    def transform(self, evaluator, literal, examples):
        clause = HornClause(
            self.initial_clause.head, *list(self.initial_clause.body))
        clause.body.append(literal)
        for clause_modifier in self.clause_modifiers:
            clause = clause_modifier.modify_clause(clause, examples)
        evaluator.horn_clause = clause

        return evaluator

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
            if isinstance(ClauseModifier, value):
                self._clause_modifiers = [value]
            else:
                self._clause_modifiers = list(value)
