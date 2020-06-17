"""
Handles the transformation of the evaluation object into async theory
evaluator.
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.knowledge.examples import Examples
from src.util.multiprocessing.theory_evaluation import AsyncTheoryEvaluator

V = TypeVar('V')
E = TypeVar('E')


# TODO: implement
class EquivalentHornClause:
    """
    A container for a Horn clause with improved information about it
    generation, in order to detect equivalent clauses and make atoms relevant
    to the equivalent clauses also relevant to this one.
    """

    def __init__(self, head):
        self.head = head
        self.horn_clause = None
        self.clause_body = None
        self.substitution_maps = None
        self.last_literal = None

    def build_appended_candidates(self, candidates):
        pass

    def build_initial_clause_candidates(self, candidates, skip_atom,
                                        skip_clause):
        pass


class EquivalentClauseAtom:
    """
    Clause to extend the concept of equivalent atom, also comparing the
    clause's body
    """


class AsyncEvaluationTransformer(ABC, Generic[V, E]):
    """
    Class to encapsulate instances of type `TV` into `AsyncTheoryEvaluator`s.
    """

    @abstractmethod
    def transform(self, evaluator, v, examples):
        """
        Transforms a element of type `V` into an `AsyncTheoryEvaluator`.

        :param evaluator: the evaluator
        :type evaluator: AsyncTheoryEvaluator[E]
        :param v: the element
        :type v: V
        :param examples: the examples
        :type examples: Examples
        :return: the async theory evaluator
        :rtype: AsyncTheoryEvaluator[E]
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
