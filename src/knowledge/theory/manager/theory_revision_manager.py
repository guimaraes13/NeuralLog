"""
Manages the revision of the theory.
"""
import logging

from src.knowledge.theory.evaluation.metric.theory_metric import TheoryMetric, \
    RocCurveMetric
from src.knowledge.theory.manager.revision.revision_examples import \
    RevisionExamples
from src.knowledge.theory.manager.revision.revision_manager import \
    RevisionManager
from src.knowledge.theory.manager.revision.revision_operator_evaluator import \
    RevisionOperatorEvaluator
from src.knowledge.theory.manager.revision.revision_operator_selector import \
    RevisionOperatorSelector
from src.structure_learning.structure_learning_system import \
    StructureLearningSystem
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

    def __init__(self, learning_system=None, revision_manager=None,
                 theory_metric=None, train_using_all_examples=True):
        """
        Creates a theory revision manager.

        :param learning_system: the learning system
        :type learning_system: StructureLearningSystem
        :param revision_manager: the revision manager
        :type revision_manager: RevisionManager
        :param theory_metric: the metric to optimize
        :type theory_metric: TheoryMetric
        :param train_using_all_examples: if `True`, the model will use all
        the available examples, in the revision examples, for training/learning;
        otherwise, only the relevant examples will be used
        :type train_using_all_examples: bool
        """
        self.learning_system = learning_system
        self.revision_manager = revision_manager
        self.theory_metric = theory_metric
        self.train_using_all_examples = train_using_all_examples
        self.last_theory_change = 0.0
        self.theory_evaluation = 0.0

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        if self.theory_metric is None:
            self.theory_metric = DEFAULT_THEORY_METRIC()
        super().initialize()

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["learning_system", "revision_manager", "theory_metric"]

    def revise(self, revision_examples):
        """
        Method to trigger the revision of the theory.

        :param revision_examples: the revision examples
        :type revision_examples:
            RevisionExamples or collections.Iterable[RevisionExamples]
        """
        self.revision_manager.revise(
            revision_examples, self.train_using_all_examples)

    def apply_revision(self, operator_selector, examples):
        """
        Uses the `operator_selector` to computes the possible revision on the
        theory, based on the examples. Then, if a revision meets
        the accepting criteria, it is applied to the current theory.

        :param operator_selector: the operator selector
        :type operator_selector: RevisionOperatorSelector
        :param examples: the revision examples
        :type examples: RevisionExamples
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
            self.theory_metric)
        logger.debug("Operator selected for revision:\t%s", operator_evaluator)
        if operator_selector is None:
            return False

        return self.apply_operator(
            operator_evaluator, examples, NO_IMPROVEMENT_THRESHOLD)

    def evaluate_current_theory(self, examples):
        """
        Evaluates the current theory, based on the examples.

        :param examples: the examples
        :type examples: RevisionExamples
        :return: the evaluation of the theory
        :rtype: float
        """
        return self.theory_metric.evaluate(examples.relevant_examples,
                                           examples.get_inferred_examples(
                                               self.last_theory_change))

    def apply_operator(self, operator_evaluator, examples,
                       improvement_threshold):
        """
        Compares the new proposed theory against the current one. If the new
        theory outperforms the current theory by the given threshold,
        the revision is applied.

        :param operator_evaluator: the operator evaluator
        :type operator_evaluator: RevisionOperatorEvaluator
        :param examples: the examples
        :type examples: RevisionExamples
        :param improvement_threshold: the improvement threshold
        :type improvement_threshold: float
        :return: `True`, if the call result in an actual change of the
        theory; otherwise, `False`
        :rtype: bool
        """
        revised_metric = operator_evaluator.evaluate_operator(
            examples.relevant_examples, self.theory_metric)
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
                operator_evaluator.theory_revision_accepted(revised_theory)
                log_message = "Theory modification accepted. Improvement of " \
                              "%d, over %d, threshold of %d."
                self.last_theory_change = time_measure.performance_time()
                theory_changed = True
        logger.debug(log_message, improvement,
                     self.theory_evaluation, improvement_threshold)
        logger.debug(THEORY_CONTENT_MESSAGE, self.learning_system.theory)
        return theory_changed
