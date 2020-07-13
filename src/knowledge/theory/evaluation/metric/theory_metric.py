"""
The metrics to evaluate the system on examples.
"""

from typing import TypeVar, Generic, List, Tuple, Dict, Any

import sklearn

"""
Handles the theory metrics.
"""
import math
import sys
from abc import abstractmethod

from src.knowledge.examples import Examples, ExamplesInferences
from src.language.language import Atom
from src.util import Initializable


class TheoryMetric(Initializable):
    """
    Class to define the theory metric.
    """

    OPTIONAL_FIELDS: Dict[str, Any] = {
        "parameters_retrain": False
    }

    def __init__(self, parameters_retrain=None):
        """
        Creates a theory metric.

        :param parameters_retrain: if `True`, the parameters will be trained
        before each candidate evaluation on this metric.
        :type parameters_retrain: Optional[bool]
        """
        self.parameters_retrain = parameters_retrain
        if parameters_retrain is None:
            self.parameters_retrain = self.OPTIONAL_FIELDS["parameters_retrain"]

    # noinspection PyMissingOrEmptyDocstring
    @property
    def default_value(self):
        return 0.0

    @abstractmethod
    def compute_metric(self, examples, inferred_values):
        """
        Evaluates the theory according to the metric.

        :param examples: the examples
        :type examples: Examples
        :param inferred_values: the inferred examples
        :type inferred_values: ExamplesInferences
        :return: the evaluation of the theory
        :rtype: float
        """
        pass

    def difference(self, candidate, current):
        """
        Calculates a quantitative difference between the candidate and the
        current evaluations.

        If the candidate is better than the current, it should return a
        positive number, representing how much it is better.

        If the candidate is worst than the current, it should return a
        negative number, representing how much it is worst.

        If they are equal, it should return 0.

        The default implementation should perform well, given an appropriated
        `compare` is implemented.

        :param candidate: the candidate evaluation
        :type candidate: float
        :param current: the current evaluation
        :type current: float
        :return: the quantitative difference between the candidate and the
        current evaluation
        :rtype: float
        """
        return math.copysign(
            candidate - current, self.compare(candidate, current))

    # noinspection PyMethodMayBeStatic
    def compare(self, o1, o2):
        """
        Compares the two objects. This comparator imposes an ordering that
        are INCONSISTENT with equals; in the sense that two different
        theories might have the same evaluation for a given knowledge base. In
        addition, two equal theories (and its parameters) MUST have the same
        evaluation for the same knowledge base.

        By default, as higher the metric, better the theory. Override this
        method, otherwise.

        :param o1: the first object to compare
        :type o1: float
        :param o2: the second object to compare
        :type o2: float
        :return: `0` if `o1` is equal to `o2`; a value less than `0` if `o1` is
        numerically less than `o2`; a value greater than `0` if `o1` is
        numerically greater than `o2`
        :rtype: int
        """
        return o1 - o2

    @abstractmethod
    def get_range(self):
        """
        Gets the range of the metric. The range of a metric is the difference
        between the maximum and the minimum value a metric can assume. If the
        range is infinite, this is, the metric is unbounded, return `None`.

        :return: the range of the metric; or `None`, if the range is infinite
        :rtype: float or None
        """
        pass

    def get_best_possible_improvement(self, current_value):
        """
        Calculates the best possible improvement over the `current_value`.
        The best possible improvement is the comparison between the maximum
        possible value of the metric against the `current_value`.

        :param current_value: the current value
        :type current_value: float
        :return: the maximum possible improvement over the current value
        :rtype: float
        """
        return self.difference(self.get_maximum_value(), current_value)

    @abstractmethod
    def get_maximum_value(self):
        """
        Gets the maximum value of the metric. The maximum value of a metric
        is the best value the metric can assume.

        :return: the maximum value of the metric
        :rtype: float
        """
        pass

    def __eq__(self, other):
        if id(self) == id(other):
            return True

        if not isinstance(other, TheoryMetric):
            return False

        if self.__class__.__name__ != other.__class__.__name__:
            return False

        if self.parameters_retrain != other.parameters_retrain:
            return False

        return self.default_value == other.default_value

    def __hash__(self):
        return hash(
            (self.parameters_retrain, self.default_value))

    @abstractmethod
    def __repr__(self):
        pass


J = TypeVar('J')  # Key type.
K = TypeVar('K')  # Value type.


class AccumulatorMetric(TheoryMetric, Generic[J, K]):
    """
    A template for metrics that simply accumulates a value for each proved
    example.

    J: The type of the initial value.
    K: The type of the value that is accumulated to the initial value.
    """

    OPTIONAL_FIELDS = TheoryMetric.OPTIONAL_FIELDS
    OPTIONAL_FIELDS.update({
        "ABSENT_PREDICTION_VALUE": 0.0
    })

    def __init__(self):
        """
        Creates an accumulator metric.
        """
        super().__init__()
        self.ABSENT_PREDICTION_VALUE = \
            self.OPTIONAL_FIELDS["ABSENT_PREDICTION_VALUE"]

    # noinspection PyMissingOrEmptyDocstring
    def compute_metric(self, examples, inferred_values):
        evaluation = self.calculate_evaluation(examples, inferred_values)
        if evaluation is None:
            return self.default_value
        else:
            return self.calculate_result(evaluation)

    def calculate_evaluation(self, examples, inferred_values):
        """
        Calculates the internal evaluation of the inferred results over the
        examples.

        :param examples: the examples
        :type examples: Examples
        :param inferred_values: the inferred values
        :type inferred_values: ExamplesInferences
        :return: the accumulation of all the results
        :rtype: J or None
        """
        if inferred_values is None:
            inferred_values = ExamplesInferences()

        result: J = self.initial_value()
        for p, facts in examples.items():
            inferred = inferred_values.get(p, dict())
            for key, atom in facts.items():
                prediction = inferred.get(key)
                example_value: K = self.calculate_value(atom, prediction)
                result = self.accumulate_value(result, example_value)

        return result

    @abstractmethod
    def initial_value(self):
        """
        Gets the initial value. This value bust be the neutral element of the
        `self.accumulate_value` method.

        :return: the initial value
        :rtype: J
        """
        pass

    @abstractmethod
    def accumulate_value(self, current, new):
        """
        Accumulates the `new` value into the `current` one.

        :param current: the current value
        :type current: J
        :param new: the new value
        :type new: K
        :return: the accumulates value
        :rtype: J
        """
        pass

    @abstractmethod
    def calculate_value(self, example, prediction):
        """
        Calculates the value that must be accumulated, based on the example
        and its prediction.

        :param example: the example
        :type example: Atom
        :param prediction: the prediction
        :type prediction: float or None
        :return: the value that must be accumulated
        :rtype: K
        """
        pass

    @abstractmethod
    def calculate_result(self, result):
        """
        Calculates the final metric result, based on the accumulated result.

        :param result: the accumulated result
        :type result: J
        :return: the final metric result
        :rtype: float
        """
        pass


class ListAccumulator(AccumulatorMetric[Tuple[List[float], List[float]],
                                        Tuple[float, float]]):
    """
    Accumulates the true and prediction values in two lists in order to
    compute curve metrics.
    """

    OPTIONAL_FIELDS = AccumulatorMetric.OPTIONAL_FIELDS
    OPTIONAL_FIELDS.update({
        "ABSENT_PREDICTION_VALUE": -sys.float_info.max
    })

    def __init__(self):
        """
        Creates a list accumulator.
        """
        super().__init__()
        self.ABSENT_PREDICTION_VALUE = \
            self.OPTIONAL_FIELDS["ABSENT_PREDICTION_VALUE"]
        "Value to be used in the absence of the prediction of an example."

    # noinspection PyMissingOrEmptyDocstring
    def get_range(self):
        return 1.0

    # noinspection PyMissingOrEmptyDocstring
    def get_maximum_value(self):
        return 1.0

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return []

    # noinspection PyMissingOrEmptyDocstring
    def initial_value(self):
        return [], []

    # noinspection PyMissingOrEmptyDocstring
    def accumulate_value(self, current, new):
        current[0].append(new[0])
        current[1].append(new[1])

        return current

    # noinspection PyMissingOrEmptyDocstring
    def calculate_value(self, example, prediction):
        if prediction is None:
            prediction = self.ABSENT_PREDICTION_VALUE

        return example.weight, prediction

    # noinspection PyMissingOrEmptyDocstring
    @abstractmethod
    def calculate_result(self, result):
        pass


def append_default_values(*results):
    """
    Appends a positive and a negative example to results in order to
    avoid the error of the curve not being defined for only one class.

    :param results: the true and predicted results
    :type results: List
    """
    for result in results:
        result.append(0.0)
        result.append(1.0)


class RocCurveMetric(ListAccumulator):
    """
    Computes the area under the ROC curve.
    """

    # noinspection PyMissingOrEmptyDocstring
    def calculate_result(self, result):
        append_default_values(*result)
        return sklearn.metrics.roc_auc_score(*result)

    def __repr__(self):
        return "ROC Curve"


class PrecisionRecallCurveMetric(ListAccumulator):
    """
    Computes the area under the Precision-Recall curve.
    """

    # noinspection PyMissingOrEmptyDocstring
    def calculate_result(self, result):
        return sklearn.metrics.average_precision_score(*result)

    def __repr__(self):
        return "PR Curve"


class LikelihoodMetric(AccumulatorMetric[float, float]):
    """
    Computes the likelihood of a theory, given the examples.

    This class assumes that the predictions are probabilities in the
    [0, 1] real interval.

    The likelihood of a theory is measured by the product of the probability
    of each example, given the theory. If the example is negative,
    the complement of the probability of the example is used, instead.
    """

    EPSILON = 1.0e-4
    """
    The minimal value to be multiplied with the result. This prevents the 
    that probability goes to `0` if a positive example is not proved or a 
    negative one gets probability of `1`.
    
    This is also issued to check if an example is positive. An example is 
    positive if its weight is less than EPSILON apart to `1`.
    """

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return []

    # noinspection PyMissingOrEmptyDocstring
    def get_range(self):
        return 1.0

    # noinspection PyMissingOrEmptyDocstring
    def get_maximum_value(self):
        return 1.0

    # noinspection PyMissingOrEmptyDocstring
    def initial_value(self):
        return 1.0

    # noinspection PyMissingOrEmptyDocstring
    def accumulate_value(self, current, new):
        return current * new

    # noinspection PyMissingOrEmptyDocstring
    def calculate_value(self, example, prediction):
        if prediction is None:
            prediction = 0.0
        else:
            prediction = max(min(0.0, prediction), 1.0)
        if abs(example.weight - 1.0) > self.EPSILON:
            prediction = 1.0 - prediction
        return max(prediction, self.EPSILON)

    # noinspection PyMissingOrEmptyDocstring
    def calculate_result(self, result):
        return result

    def __repr__(self):
        return "Likelihood"


class LogLikelihoodMetric(LikelihoodMetric):
    """
    Computes the log likelihood of a theory, given the examples.

    This class assumes that the predictions are probabilities in the
    [0, 1] real interval.

    The likelihood of a theory is measured by the sum of the log of the
    probability of each example, given the theory. If the example is negative,
    the complement of the probability of the example is used, instead.
    """

    def __init__(self):
        """
        Creates a log likelihood metric.
        """
        super().__init__()

    # noinspection PyMissingOrEmptyDocstring
    @property
    def default_value(self):
        return -sys.float_info.max

    # noinspection PyMissingOrEmptyDocstring
    def get_range(self):
        return None

    # noinspection PyMissingOrEmptyDocstring
    def get_maximum_value(self):
        return 0.0

    # noinspection PyMissingOrEmptyDocstring
    def initial_value(self):
        return 0.0

    # noinspection PyMissingOrEmptyDocstring
    def accumulate_value(self, current, new):
        return max(current + new, self.default_value)

    # noinspection PyMissingOrEmptyDocstring
    def calculate_value(self, example, prediction):
        return math.log(super().calculate_value(example, prediction))

    def __repr__(self):
        return "Log Likelihood"
