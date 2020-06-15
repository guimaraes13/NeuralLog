"""
Handles parallel executions.
"""
from typing import TypeVar, Generic

from src.util.multiprocessing.evaluation_transformer import \
    AsyncEvaluationTransformer

V = TypeVar('V')
E = TypeVar('E')


# TODO: implement
class MultiprocessingEvaluation(Generic[V, E]):
    """
    Class to handle the multiprocessing evaluation of candidate revisions.
    """

    def __init__(self, learning_system, theory_metric, evaluation_timeout,
                 transformer, number_of_process=1):
        """
        Creates a multiprocessing evaluation.

        :param learning_system: the learning system
        :type learning_system: StructureLearningSystem
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
