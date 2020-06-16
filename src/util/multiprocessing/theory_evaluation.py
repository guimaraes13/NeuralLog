"""
Handles the evaluation of theories, in parallel.
"""
import logging
import multiprocessing
import sys
from multiprocessing import Queue
from multiprocessing.context import Process
from typing import TypeVar, Generic, Optional

from src.util import time_measure

logger = logging.getLogger(__name__)
E = TypeVar('E')


class ClauseEvaluationProcess(Process):
    """
    Class to represent an evaluation process.
    """

    def __init__(self, theory_evaluator, theory_metric, examples, horn_clause,
                 returning_queue):
        """
        Creates a clause evaluation process.

        :param theory_evaluator: the theory evaluator
        :type theory_evaluator: TheoryEvaluator
        :param theory_metric: the theory metric
        :type theory_metric: TheoryMetric
        :param examples: the examples
        :type examples: Examples
        :param horn_clause: the Horn clause
        :type horn_clause: HornClause
        :param returning_queue: a queue to return the value
        :type returning_queue: Queue
        """
        super().__init__(target=self)
        self.theory_evaluator = theory_evaluator
        self.theory_metric = theory_metric
        self.examples = examples
        self.horn_clause = horn_clause

        self.evaluation = None

        self.returning_queue = returning_queue

    def __call__(self, *args, **kwargs):
        self.evaluation_finished = False
        self.evaluation_time = sys.float_info.max
        self.begin_performance = time_measure.performance_time()
        self.begin_real = time_measure.process_time()
        self.evaluation = \
            self.theory_evaluator.evaluate_theory_appending_clause(
                self.examples, self.theory_metric, [self.horn_clause]
            )
        self.end_real = time_measure.process_time()
        self.end_performance = time_measure.performance_time()
        self.returning_queue.put(self)


class AsyncTheoryEvaluator(Generic[E]):
    """
    Handles an asynchronous evaluation of a theory.

    It specifies a maximum amount of time the evaluation of the theory must
    take.
    """

    def __init__(self, examples, theory_evaluator, theory_metric,
                 timeout=None, horn_clause=None, element=None):
        """
        Creates an async theory evaluator.

        :param examples: the examples
        :type examples: Examples
        :param theory_evaluator: the theory evaluator
        :type theory_evaluator: TheoryEvaluator
        :param theory_metric: the theory metric
        :type theory_metric: TheoryMetric
        :param timeout: the timeout
        :type timeout: int or None
        :param horn_clause: the Horn clause
        :type horn_clause: HornClause
        :param element: an element
        :type element: E
        """
        self.examples = examples
        "The examples."

        self.theory_evaluator = theory_evaluator
        "The theory evaluator."

        self.theory_metric = theory_metric
        "The theory metric."

        self.timeout = timeout
        "The timeout."

        self.horn_clause = horn_clause
        "The Horn clause."

        self.element = element
        "An element"

        self.has_finished = False
        "If the process has successfully finished."

        self.evaluation: Optional[float] = None
        "The evaluation of the clause."

        self.real_time: Optional[float] = None
        "The real time take to evaluate the clause"

        self.performance_time: Optional[float] = None
        "The process time take to evaluate the clause"

    def __call__(self, *args, **kwargs):
        manager = multiprocessing.Manager()
        queue = manager.Queue(1)
        process = ClauseEvaluationProcess(
            self.theory_evaluator, self.theory_metric, self.examples,
            self.horn_clause, queue)
        # noinspection PyBroadException
        try:
            process.start()
            process.join(self.timeout)
        except Exception:
            logger.exception("Error evaluating the candidate theory, reason:")
        if process.is_alive():
            process.kill()
            process.close()
            logger.debug(
                "Evaluation of the theory timed out after %d seconds.",
                self.timeout)
        else:
            result = queue.get()
            self.has_finished = True
            self.evaluation = result.evaluation
            self.real_time = result.end_real - result.begin_real
            self.performance_time = \
                result.end_performance - result.begin_performance

        return self

    def __repr__(self):
        return "[{}]: {}".format(self.__class__.__name__, self.horn_clause)

    def _get_key(self):
        return self.horn_clause, self.has_finished, self.evaluation, \
               self.real_time, self.performance_time

    def __hash__(self):
        return hash(self._get_key())

    def __eq__(self, other):
        if not isinstance(other, AsyncTheoryEvaluator):
            return False

        return self._get_key() == other._get_key()
