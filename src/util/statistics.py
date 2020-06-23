"""
Measures training statistics.
"""
from functools import reduce
from typing import Optional, Dict, List

from src.knowledge.theory.evaluation.metric.theory_metric import TheoryMetric
from src.util.time_measure import TimeMeasure


def get_space(metric_name, maximum_metric_size):
    """
    Gets the space to create a tabulation.

    :param metric_name: the metric name
    :type metric_name: str
    :param maximum_metric_size: the maximum size of the metric name
    :type maximum_metric_size: int
    :return: the spaces to be appended to the message
    :rtype: str
    """
    return " " * (maximum_metric_size - len(metric_name) + 1)


class RunStatistics:
    """
    Class to save statistics about a run of the learning method.
    """

    def __init__(self, knowledge_base_size=0, train_size=0, test_size=0):
        """
        Creates a run statistic.

        :param knowledge_base_size: the size of the knowledge base
        :type knowledge_base_size: int
        :param train_size: the number of train examples
        :type train_size: int
        :param test_size: the number of test examples
        :type test_size: int
        """
        self.time_measure: Optional[TimeMeasure] = None

        self.knowledge_base_size = knowledge_base_size
        self.train_size = train_size
        self.test_size = test_size

        self.train_evaluation: Optional[Dict[TheoryMetric, float]] = None
        self.test_evaluation: Optional[Dict[TheoryMetric, float]] = None

        self.maximum_metric_size = 0

    def get_sorted_metrics(self):
        """
        Gets the metrics sorted alphabetically.

        :return: the metrics
        :rtype: List[TheoryMetric]
        """
        metric_set = set()
        if self.train_evaluation:
            metric_set.update(self.train_evaluation.keys())
        if self.test_size:
            metric_set.update(self.test_evaluation.keys())
        self.maximum_metric_size = reduce(
            lambda x, y: max(x, y), map(lambda x: len(str(x)), metric_set), 0)
        return list(sorted(metric_set, key=lambda x: str(x)))

    def append_evaluation(self, sorted_metrics, message, evaluation, label):
        """
        Appends the evaluation of the run to the `message`.

        :param sorted_metrics: the sorted metrics
        :type sorted_metrics: List[TheoryMetric]
        :param message: the message
        :type message: str
        :param evaluation: the evaluation
        :type evaluation: Dict[TheoryMetric, float]
        :param label: the label of the evaluation
        :type label: str
        :return: the message
        :rtype: str
        """
        message += f"\t- {label} Evaluation:\n"
        for key in sorted_metrics:
            value = evaluation.get(key)
            if value is not None:
                metric_name = str(key).strip()
                message += f"\t\t- {metric_name}:"
                message += \
                    f"{get_space(metric_name, self.maximum_metric_size)}"
                message += f"{value}\n"

        return message

    def __repr__(self):
        sorted_metric = self.get_sorted_metrics()
        message = "Run Statistics:\n"
        message += f"\t- Knowledge Size:\t{self.knowledge_base_size}\n"
        message += f"\t- Examples Size: \t{self.train_size}\n"

        if self.test_evaluation:
            message += f"\t- Test Size: \t\t{self.test_size}\n"

        if self.train_evaluation:
            message = self.append_evaluation(
                sorted_metric, message, self.train_evaluation, "Train")
        if self.test_evaluation:
            message = self.append_evaluation(
                sorted_metric, message, self.test_evaluation, "Test")

        message += "\n"
        message += f"\tTotal Run Time:\t{self.time_measure}\n"

        return message
