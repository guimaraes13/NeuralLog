"""
Handles the transformation of the evaluation object into async theory
evaluator.
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.knowledge.examples import Examples
from src.language.equivalent_clauses import EquivalentHornClause
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
