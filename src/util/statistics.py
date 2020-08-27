"""
Measures training statistics.
"""
from functools import reduce
from typing import Optional, Dict, List, TypeVar, Generic

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


T = TypeVar('T')


class IterationStatistics(Generic[T]):
    """
    Class to hold the statistics of an iteration experiment. The idea is to
    serialize this class using the yaml library, in order to be able to consult
    the statistics of a run in a way that is both human and machine friendly.
    """

    def __init__(self, number_of_iterations=None):
        self.number_of_iterations: Optional[float] = number_of_iterations
        self.iteration_names = []

        self.iteration_knowledge_sizes = []
        self.iteration_examples_sizes = []

        self.iteration_train_evaluation: List[Dict[TheoryMetric, float]] = []
        self.iteration_test_evaluation: List[Dict[TheoryMetric, float]] = []

        self.time_measure: TimeMeasure[T] = TimeMeasure()

    def add_iteration_train_evaluation(self, evaluations):
        """
        Adds the iteration train evaluations to the statistics.

        :param evaluations: the evaluations
        :type evaluations: Dict[TheoryMetric, float]
        """
        self.iteration_train_evaluation.append(evaluations)

    def add_iteration_test_evaluation(self, evaluations):
        """
        Adds the iteration test evaluations to the statistics.

        :param evaluations: the evaluations
        :type evaluations: Dict[TheoryMetric, float]
        """
        self.iteration_test_evaluation.append(evaluations)

    def get_sorted_metrics(self):
        """
        Gets the metrics of the evaluations, sorted.

        :return: the metrics of the evaluations
        :rtype: List[TheoryMetric]
        """
        metric_set = set()
        for evaluation in self.iteration_train_evaluation:
            metric_set.update(evaluation.keys())
        for evaluation in self.iteration_test_evaluation:
            metric_set.update(evaluation.keys())

        metrics = list(metric_set)
        metrics.sort(key=lambda x: str(x))

        return metrics

    def append_general_information(self, description):
        """
        Appends the general information to the description.

        :param description: the current description
        :type description: str
        :return: the appended description
        :rtype: str
        """

        description += self.__class__.__name__
        description += "\n"

        description += f"\tNumber of iterations:\t{self.number_of_iterations}\n"

        return description

    def append_knowledge_size(self, description, total_size, index):
        """
        Appends the knowledge size of the iteration to the description.

        :param description: the current description
        :type description: str
        :param total_size: the total size until the index, included
        :type total_size: int
        :param index: the current index
        :type index: int
        :return: the appended description
        :rtype: str
        """
        description += f"\t\t\t- New knowledge of iteration:" \
                       f"\t{self.iteration_names[index]}:\t"
        description += f"{self.iteration_knowledge_sizes[index]}\n"
        description += f"\t\t\t- Iteration knowledge size:\t{total_size}\n"

        return description

    def append_examples_size(self, description, total_size, index):
        """
        Appends the examples size of the iteration to the description.

        :param description: the current description
        :type description: str
        :param total_size: the total size until the index, included
        :type total_size: int
        :param index: the current index
        :type index: int
        :return: the appended description
        :rtype: str
        """
        description += f"\t\t\t- New examples of iteration:" \
                       f"\t{self.iteration_names[index]}:\t"
        description += f"{self.iteration_examples_sizes[index]}\n"
        description += f"\t\t\t- Iteration examples size:\t{total_size}\n"

        return description

    @staticmethod
    def append_evaluation(description, sorted_metrics, index,
                          iteration_evaluation, label):
        """
        Appends the examples size of the iteration to the description.

        :param description: the current description
        :type description: str
        :param sorted_metrics: the sorted metrics
        :type sorted_metrics: List[TheoryMetric]
        :param index: the current index
        :type index: int
        :param iteration_evaluation: the evaluation of the iteration
        :type iteration_evaluation: List[Dict[TheoryMetric, float]]
        :param label: the label of the evaluation
        :type label: str
        :return: the appended description
        :rtype: str
        """
        has_metric = False
        if index < len(iteration_evaluation):
            description += f"\t\t\t- {label} evaluation:\n"
            for metric in sorted_metrics:
                value = iteration_evaluation[index].get(metric)
                if value is not None:
                    has_metric = True
                    description += f"\t\t\t\t- {metric}:\t{value}\n"
            if not has_metric:
                description += f"\t\t\t\t- NO EVALUATION\n"
        return description

    def __repr__(self):
        sorted_metrics = self.get_sorted_metrics()
        description = ""

        description = self.append_general_information(description)

        description += "\tInfo by Iteration:\n"
        total_knowledge_size = 0
        total_examples_size = 0
        for i in range(len(self.iteration_names)):
            iteration_name = self.iteration_names[i]
            total_knowledge_size += self.iteration_knowledge_sizes[i]
            total_examples_size += self.iteration_examples_sizes[i]

            description += f"\t\tIteration:\t{iteration_name}\n"

            description = \
                self.append_knowledge_size(description, total_knowledge_size, i)
            description = \
                self.append_examples_size(description, total_examples_size, i)

            description = self.append_evaluation(
                description, sorted_metrics, i,
                self.iteration_train_evaluation, "Train")
            description = self.append_evaluation(
                description, sorted_metrics, i,
                self.iteration_test_evaluation, "Test")

            description += "\n"

        description += f"\tTotal run time:\t{self.time_measure}\n"

        return description
