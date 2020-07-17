"""
Handles parallel executions.
"""
import collections
import logging
# noinspection PyProtectedMember
from concurrent.futures._base import Future, CancelledError
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from typing import TypeVar, Generic, Dict, Set

import src.structure_learning.structure_learning_system as sls
from src.knowledge.examples import Examples
from src.knowledge.theory.evaluation.metric.theory_metric import TheoryMetric
from src.language.language import HornClause
from src.util import OrderedSet
from src.util.multiprocessing.evaluation_transformer import \
    AsyncEvaluationTransformer
from src.util.multiprocessing.theory_evaluation import AsyncTheoryEvaluator, \
    SyncTheoryEvaluator

DEFAULT_NUMBER_OF_PROCESS = 1
DEFAULT_EVALUATION_TIMEOUT = 300

V = TypeVar('V')
E = TypeVar('E')

logger = logging.getLogger(__name__)


class MultiprocessingEvaluation(Generic[V, E]):
    """
    Class to handle the multiprocessing evaluation of candidate revisions.
    """

    def __init__(self, learning_system, theory_metric,
                 transformer, evaluation_timeout=DEFAULT_EVALUATION_TIMEOUT,
                 number_of_process=DEFAULT_NUMBER_OF_PROCESS):
        """
        Creates a multiprocessing evaluation.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param theory_metric: the theory metric
        :type theory_metric: TheoryMetric
        :param evaluation_timeout: the individual evaluation timeout, in seconds
        :type evaluation_timeout: int
        :param transformer: the evaluation transformer
        :type transformer: AsyncEvaluationTransformer[V, E]
        """

        self.learning_system = learning_system
        "The learning system."

        self.theory_metric = theory_metric
        "The theory metric."

        self.evaluation_timeout = evaluation_timeout
        "The individual evaluation timeout."

        self.transformer = transformer
        "The async evaluation transformer."

        self.number_of_process = number_of_process
        "The maximum number of process this class is allowed to create."

    def get_best_clause_from_candidates(
            self, candidates, examples, evaluation_map=None):
        """
        Evaluates the candidate clauses against the metric and returns the
        best evaluated clause.

        :param candidates: the candidates
        :type candidates: collections.Collection[V]
        :param examples: the examples
        :type examples: Examples
        :param evaluation_map: the map of rules and their evaluations
        :type evaluation_map: Dict[AsyncTheoryEvaluator[E], float] or None
        :return: the async theory evaluator containing the best evaluated clause
        :rtype: AsyncTheoryEvaluator[E]
        """
        if not candidates:
            return None

        best_clause = None
        # noinspection PyBroadException
        try:
            if evaluation_map is None:
                evaluation_map = dict()
            # best_clause = self._evaluate_using_pool(
            #     candidates, examples, evaluation_map)
            best_clause = self._evaluate_sequentially(
                candidates, examples, evaluation_map)
            if logger.isEnabledFor(logging.DEBUG):
                sorted_clauses = sorted(
                    evaluation_map.items(), key=lambda x: -x[1])
                for clause, evaluation in sorted_clauses:
                    logger.debug(
                        "Evaluation: %.3f\twith time: %.3fs\tfor rule:\t%s",
                        evaluation, clause.real_time, clause.horn_clause
                    )
        except Exception:
            logger.exception("Error when evaluating the clause, reason:")

        return best_clause

    # noinspection DuplicatedCode
    def _evaluate_sequentially(self, candidates, examples, evaluation_map=None):
        """
        Evaluates the candidates sequentially.

        :param candidates: the candidates
        :type candidates: collections.Collection[V]
        :param examples: the examples
        :type examples: Examples
        :param evaluation_map: the map of rules and their evaluations
        :type evaluation_map: Dict[AsyncTheoryEvaluator[E], float] or
        None
        :return: the async theory evaluator containing the best
        evaluated clause
        :rtype: AsyncTheoryEvaluator[E]
        """
        best_evaluation = self.theory_metric.default_value
        best_evaluator = None
        current_clause = None
        successful_runs = 0
        logger.info("[ BEGIN ]\tSequential evaluation of %d candidate(s).",
                    len(candidates))
        for candidate in candidates:
            # noinspection PyBroadException
            try:
                evaluator = SyncTheoryEvaluator(
                    examples, self.learning_system.theory_evaluator,
                    self.theory_metric, self.evaluation_timeout)
                evaluator = \
                    self.transformer.transform(evaluator, candidate, examples)
                current_clause = evaluator.horn_clause
                logger.info("Evaluating candidate:\t%s", current_clause)
                async_evaluator: AsyncTheoryEvaluator = evaluator()
                if not async_evaluator.has_finished:
                    continue
                successful_runs += 1

                evaluation = async_evaluator.evaluation
                if evaluation_map is not None:
                    evaluation_map[async_evaluator] = evaluation

                if best_evaluator is None or self.theory_metric.compare(
                        evaluation, best_evaluation) > 0.0:
                    best_evaluation = evaluation
                    best_evaluator = async_evaluator
            except (CancelledError, TimeoutError):
                logger.exception(
                    "Evaluation of the theory timed out after %d seconds.",
                    self.evaluation_timeout)
                logger.warning("Clause:\t%s", current_clause)
            except Exception:
                logger.exception("Error when evaluating the clause, reason:")
                logger.warning("Clause:\t%s", current_clause)
        logger.info("[  END  ]\tSequential evaluation.")
        return best_evaluator

    def _evaluate_using_pool(self, candidates, examples, evaluation_map=None):
        """
        Evaluates the candidates using a pool of workers.

        :param candidates: the candidates
        :type candidates: collections.Collection[V]
        :param examples: the examples
        :type examples: Examples
        :param evaluation_map: the map of rules and their evaluations
        :type evaluation_map: Dict[AsyncTheoryEvaluator[E], float] or None
        :return: the async theory evaluator containing the best evaluated clause
        :rtype: AsyncTheoryEvaluator[E]
        """
        logger.info("[ BEGIN ]\tAsynchronous evaluation of %d candidate(s).",
                    len(candidates))
        number_of_process = min(self.number_of_process, len(candidates))
        pool = ThreadPoolExecutor(number_of_process)
        futures = self.submit_candidates(candidates, examples, pool)
        pool.shutdown(True)
        logger.info("[  END  ]\tAsynchronous evaluation.")
        best_clause = \
            self.retrieve_evaluated_candidates(futures, evaluation_map)

        return best_clause

    def submit_candidates(self, candidates, examples, pool):
        """
        Submits the `candidates` to the evaluation `pool`, returning a set of
        future objects.

        :param candidates: the candidates
        :type candidates: collections.Collection[V]
        :param examples: the examples
        :type examples: Examples
        :param pool: the evaluation pool
        :type pool: ProcessPoolExecutor or ThreadPoolExecutor
        :return: the set of futures
        :rtype: Set[Future[AsyncTheoryEvaluator[E]]]
        """
        futures: Set[Future[AsyncTheoryEvaluator[E]]] = OrderedSet()
        for candidate in candidates:
            logger.debug("Submitting candidate:\t%s", candidate)
            evaluator = SyncTheoryEvaluator(
                examples, self.learning_system.theory_evaluator,
                self.theory_metric, self.evaluation_timeout)
            evaluator = \
                self.transformer.transform(evaluator, candidate, examples)
            future = pool.submit(evaluator)
            futures.add(future)

        return futures

    # noinspection DuplicatedCode
    def retrieve_evaluated_candidates(self, futures, evaluation_map=None):
        """
        Retrieves the evaluations from the `Future` `AsyncTheoryEvaluator`s
        and appends it to the `evaluation_map`, if it is not None. Also, returns
        the best evaluated Horn clause.

        :param futures: the futures
        :type futures: Set[Future[AsyncTheoryEvaluator[E]]]
        :param evaluation_map: the evaluation map
        :type evaluation_map: Dict[AsyncTheoryEvaluator[E], float] or None
        :return: the best Horn clause
        :rtype: HornClause
        """
        best_evaluation = self.theory_metric.default_value
        best_evaluator = None
        successful_runs = 0
        for future in futures:
            # noinspection PyBroadException
            try:
                async_evaluator = future.result(0)
                if not async_evaluator.has_finished:
                    continue
                successful_runs += 1

                evaluation = async_evaluator.evaluation
                if evaluation_map is not None:
                    evaluation_map[async_evaluator] = evaluation

                if best_evaluator is None or self.theory_metric.compare(
                        evaluation, best_evaluation) > 0.0:
                    best_evaluation = evaluation
                    best_evaluator = async_evaluator
            except (CancelledError, TimeoutError):
                logger.exception(
                    "Evaluation of the theory timed out after %d seconds.",
                    self.evaluation_timeout)
            except Exception:
                logger.exception("Error when evaluating the clause, reason:")

        return best_evaluator
