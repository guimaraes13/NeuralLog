"""
Manages the revision of the theory.
"""
import logging
import math
import sys

import src.knowledge.theory.manager.revision.revision_examples as revision
import src.structure_learning.structure_learning_system as sls
from src.knowledge.theory.evaluation.metric.theory_metric import TheoryMetric, \
    RocCurveMetric
from src.knowledge.theory.manager.revision.revision_manager import \
    RevisionManager
from src.knowledge.theory.manager.revision.revision_operator_evaluator import \
    RevisionOperatorEvaluator
from src.knowledge.theory.manager.revision.revision_operator_selector import \
    RevisionOperatorSelector
from src.util import Initializable, time_measure

THEORY_CONTENT_MESSAGE = "\n------------------ THEORY -----------------\n%s" \
                         "\n------------------ THEORY -----------------"

logger = logging.getLogger(__name__)

NO_IMPROVEMENT_THRESHOLD = 0.0
DEFAULT_THEORY_METRIC = RocCurveMetric


class TheoryRevisionManager(Initializable):
    """
    Represents a theory revision manager. The theory revision manager is
    responsible for decide whether a proposed theory should replace the
    current theory of the system.
    """

    OPTIONAL_FIELDS = {
        "train_using_all_examples": True,
        "last_theory_change": -sys.float_info.max,
        "theory_evaluation": 0.0,
        "improvement_threshold": NO_IMPROVEMENT_THRESHOLD
    }

    def __init__(self, learning_system=None, revision_manager=None,
                 theory_metric=None, train_using_all_examples=None,
                 improvement_threshold=None):
        """
        Creates a theory revision manager.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param revision_manager: the revision manager
        :type revision_manager: RevisionManager
        :param theory_metric: the metric to optimize
        :type theory_metric: TheoryMetric
        :param train_using_all_examples: if `True`, the model will use all
        the available examples, in the revision examples, for training/learning;
        otherwise, only the relevant examples will be used
        :type train_using_all_examples: Optional[bool]
        """
        self.learning_system = learning_system
        self.revision_manager = revision_manager
        self.theory_metric = theory_metric
        self.train_using_all_examples = train_using_all_examples
        if train_using_all_examples is None:
            self.train_using_all_examples = \
                self.OPTIONAL_FIELDS["train_using_all_examples"]
        self.improvement_threshold = improvement_threshold
        if self.improvement_threshold is None:
            self.improvement_threshold = \
                self.OPTIONAL_FIELDS["improvement_threshold"]
        self.last_theory_change = self.OPTIONAL_FIELDS["last_theory_change"]
        self.theory_evaluation = self.OPTIONAL_FIELDS["theory_evaluation"]

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        if self.theory_metric is None:
            self.theory_metric = DEFAULT_THEORY_METRIC()
        self.theory_metric.initialize()
        self.revision_manager.theory_revision_manager = self
        self.revision_manager.initialize()

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["learning_system", "revision_manager", "theory_metric"]

    def revise(self, revision_examples):
        """
        Method to trigger the revision of the theory.

        :param revision_examples: the revision examples
        :type revision_examples: revision.RevisionExamples or
            collections.Collection[revision.RevisionExamples]
        """
        self.revision_manager.revise(revision_examples)

    def apply_revision(self, operator_selector, examples):
        """
        Uses the `operator_selector` to computes the possible revision on the
        theory, based on the examples. Then, if a revision meets
        the accepting criteria, it is applied to the current theory.

        :param operator_selector: the operator selector
        :type operator_selector: RevisionOperatorSelector
        :param examples: the revision examples
        :type examples: revision.RevisionExamples
        :raise TheoryRevisionException: if an error occur during the revision
        of the theory
        :return: `True`, if the call result in an actual change of the
        theory; otherwise, `False`
        :rtype: bool
        """
        self.theory_evaluation = self.evaluate_current_theory(examples)
        number_of_examples = examples.get_number_of_examples(
            self.train_using_all_examples)
        logger.debug("Calling the revision on %d examples.", number_of_examples)
        operator_evaluator = operator_selector.select_operator(
            examples.get_training_examples(self.train_using_all_examples),
            self.theory_metric, minimum_threshold=self.improvement_threshold)
        logger.debug("Operator selected for revision:\t%s",
                     operator_evaluator.revision_operator)
        if operator_selector is None:
            return False

        return self.apply_operator(
            operator_evaluator, examples, self.improvement_threshold)

    def evaluate_current_theory(self, examples):
        """
        Evaluates the current theory, based on the examples.

        :param examples: the examples
        :type examples: revision.RevisionExamples
        :return: the evaluation of the theory
        :rtype: float
        """
        return self.theory_metric.compute_metric(
            examples.relevant_examples,
            examples.get_inferred_values(self.last_theory_change))

    def apply_operator(self, operator_evaluator, examples,
                       improvement_threshold):
        """
        Compares the new proposed theory against the current one. If the new
        theory outperforms the current theory by the given threshold,
        the revision is applied.

        :param operator_evaluator: the operator evaluator
        :type operator_evaluator: RevisionOperatorEvaluator
        :param examples: the examples
        :type examples: revision.RevisionExamples
        :param improvement_threshold: the improvement threshold
        :type improvement_threshold: float
        :return: `True`, if the call result in an actual change of the
        theory; otherwise, `False`
        :rtype: bool
        """
        relevant_examples = examples.relevant_examples
        revised_metric = operator_evaluator.evaluate_operator(
            relevant_examples, self.theory_metric, improvement_threshold)
        logger.debug("Revised theory evaluation:\t%f", revised_metric)
        improvement = self.theory_metric.difference(
            revised_metric, self.theory_evaluation)
        log_message = "Theory modification skipped due no significant " \
                      "improvement. Improvement of %f, over %f, " \
                      "threshold of %f."
        theory_changed = False
        if improvement >= improvement_threshold:
            training_examples = examples.get_training_examples(
                self.train_using_all_examples)
            revised_theory = \
                operator_evaluator.get_revised_theory(training_examples)
            if revised_theory is not None:
                self.learning_system.theory = revised_theory
                self.learning_system.train_parameters(training_examples)
                self.learning_system.save_trained_parameters()
                operator_evaluator.theory_revision_accepted(
                    revised_theory, relevant_examples)
                log_message = "Theory modification accepted. Improvement of " \
                              "%f, over %f, threshold of %f."
                self.last_theory_change = time_measure.performance_time()
                theory_changed = True
        logger.debug(log_message, improvement,
                     self.theory_evaluation, improvement_threshold)
        logger.debug(THEORY_CONTENT_MESSAGE, self.learning_system.theory)
        return theory_changed


class HoeffdingBoundTheoryManager(TheoryRevisionManager):
    """
    Responsible to decide when to revise the theory, based on the Hoeffding's
    bound.

    The theory will be updated only if the improvement on the sample is
    greater than the Hoeffding's bound.

    For a given value `delta` in [0, 1], a metric whose range size is `R`,
    and a sample containing `n` independent examples; the Hoeffding's bound
    `epsilon` is given by the formula `epsilon = sqrt((R^2 * ln(1/delta)/(2n)).
    """

    OPTIONAL_FIELDS = dict(TheoryRevisionManager.OPTIONAL_FIELDS)
    OPTIONAL_FIELDS.update({
        "delta": 1.0e-6,
        "delta_update_function": None
    })

    def __init__(self, learning_system=None, revision_manager=None,
                 theory_metric=None, train_using_all_examples=None,
                 delta=None, delta_update_function=None):
        super().__init__(learning_system, revision_manager, theory_metric,
                         train_using_all_examples)

        self.delta = delta
        "The delta value to compute the Hoeffding's bound."

        if self.delta is None:
            self.delta = self.OPTIONAL_FIELDS["delta"]

        self.delta_update_function = delta_update_function
        """
        A function (as a string), to update the value of delta, based on itself.
        """

        self._delta_function = None

    @property
    def current_delta(self):
        """
        The delta value to compute the Hoeffding's bound.
        """
        return self.delta

    @current_delta.setter
    def current_delta(self, value):
        self.set_delta(value)

    def set_delta(self, value):
        """
        Sets the value of delta, if `value` is valid.

        It is valid if it is in [0, 1].

        :param value: the value
        :type value: float
        :return: returns `True` if `value` is valid; otherwise, `False`
        :rtype: bool
        """
        if 0.0 <= value <= 1.0:
            self.delta = value
            return True
        return False

    @staticmethod
    def _compile_delta(delta_update_function):
        """
        Compiles the delta update function.

        :param delta_update_function: the delta function
        :type delta_update_function: str
        :return: the compiled function
        :rtype: Callable[[float], float]
        """
        if delta_update_function is None or \
                not isinstance(delta_update_function, str):
            return lambda x: x
        return eval(delta_update_function, {})

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        self._delta_function = self._compile_delta(self.delta_update_function)

    # noinspection PyMissingOrEmptyDocstring
    def apply_revision(self, operator_selector, examples):
        epsilon = self.calculate_hoeffding_bound(
            self.theory_metric.get_range(), examples.relevant_examples.size())
        self.theory_evaluation = self.evaluate_current_theory(examples)
        best_possible_improvement = \
            self.theory_metric.get_best_possible_improvement(
                self.theory_evaluation)
        if best_possible_improvement < epsilon:
            logger.info(
                "Skipping the revision on examples, there are not enough "
                "examples to exceed the threshold. Current evaluation:\t%f, "
                "threshold of %f.", self.theory_evaluation, epsilon
            )
            return False

        number_of_examples = examples.get_number_of_examples(
            self.train_using_all_examples)
        logger.debug("Calling the revision on %d examples.", number_of_examples)
        operator_evaluator = operator_selector.select_operator(
            examples.get_training_examples(self.train_using_all_examples),
            self.theory_metric, minimum_threshold=epsilon)
        logger.debug("Operator selected for revision:\t%s",
                     operator_evaluator.revision_operator)
        if operator_selector is None:
            return False
        revised = self.apply_operator(operator_evaluator, examples, epsilon)
        if revised:
            self.update_delta()
        return revised

    def calculate_hoeffding_bound(self, metric_range, sample_size):
        """
        Calculates the Hoeffding's bound value of epsilon, based on the
        `metric_range` of the metric and the `sample_size`.

        :param metric_range: the range of the metric
        :type metric_range: float
        :param sample_size: the sample size
        :type sample_size: int
        :return: the Hoeffding's bound
        :rtype: float
        """
        return math.sqrt((metric_range * metric_range *
                          -math.log(self.current_delta)) / (2 * sample_size))

    def update_delta(self):
        """
        Updates the delta each time a revision is accepted, given a specified
        function f: R -> R, defined in `self.delta_update_function`.
        """
        old_delta = self.current_delta
        new_delta = self._delta_function(old_delta)
        if self.set_delta(new_delta):
            logger.debug("Delta updated from\t%f to\t%f", old_delta, new_delta)
        else:
            logger.debug("Delta value not updated because of range "
                         "constraints, current value:\t%f", old_delta)
