#  Copyright 2021 Victor Guimarães
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Handles the selection of the revision operators.
"""
import logging
from abc import abstractmethod, ABC
from collections import Collection

import neurallog.structure_learning.structure_learning_system as sls
from neurallog.knowledge.examples import Examples
from neurallog.knowledge.theory import TheoryRevisionException
from neurallog.knowledge.theory.evaluation.metric.theory_metric import \
    TheoryMetric
from neurallog.knowledge.theory.manager.revision.revision_operator_evaluator \
    import \
    RevisionOperatorEvaluator
from neurallog.util import Initializable

logger = logging.getLogger(__name__)


class RevisionOperatorSelector(Initializable):
    """
    Class responsible for selecting the best suited revision operator.
    """

    def __init__(self, learning_system=None, operator_evaluators=None):
        """
        Creates a revision operator selector.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param operator_evaluators: the operator evaluators
        :type operator_evaluators: Collection[RevisionOperatorEvaluator] or None
        """
        self.learning_system = learning_system
        self.operator_evaluators = operator_evaluators

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        for operator_evaluator in self.operator_evaluators:
            operator_evaluator.learning_system = self.learning_system
            operator_evaluator.initialize()

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["learning_system", "operator_evaluators"]

    @abstractmethod
    def select_operator(self, examples, theory_metric, minimum_threshold=None):
        """
        Selects the best operator to revise the theory, based on the examples
        and the metric.

        :param examples: the examples
        :type examples: Examples
        :param theory_metric: the theory metric
        :type theory_metric: TheoryMetric
        :param minimum_threshold: a minimum threshold to consider by the
        operator. Implementations of this class could use this threshold in
        order to improve performance by skipping evaluating candidates
        :type minimum_threshold: Optional[float]
        :return: the best revision operator
        :rtype: RevisionOperatorEvaluator
        """
        pass


class SelectFirstRevisionOperator(RevisionOperatorSelector):
    """
    Selects the first operator evaluator to revise the theory.
    """

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def initialize(self):
        super().initialize()
        self._operator = next(iter(self.operator_evaluators))

    # noinspection PyMissingOrEmptyDocstring
    def select_operator(self, examples, theory_metric, minimum_threshold=None):
        return self._operator


class BestRevisionOperatorSelector(RevisionOperatorSelector):
    """
    Selects the best possible revision operator.
    """

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def initialize(self):
        super().initialize()
        if len(self.operator_evaluators) < 2:
            self.selector: RevisionOperatorEvaluatorSelector = \
                SingleRevisionOperatorEvaluator(self.operator_evaluators)
        else:
            self.selector: RevisionOperatorEvaluatorSelector = \
                BestSelector(self.operator_evaluators)

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def select_operator(self, examples, theory_metric, minimum_threshold=None):
        return self.selector.select_operator(
            examples, theory_metric, minimum_threshold)


class RevisionOperatorEvaluatorSelector(ABC):
    """
    Class to select the proper operator, given the target examples and the
    metric.
    """

    @abstractmethod
    def select_operator(self, targets, metric, minimum_threshold=None):
        """
        Selects the proper operator, based on the target examples and the
        metric.

        :param targets: the target examples
        :type targets: Examples
        :param metric: the metric
        :type metric: TheoryMetric
        :param minimum_threshold: a minimum threshold to consider by the
        operator. Implementations of this class could use this threshold in
        order to improve performance by skipping evaluating candidates
        :type minimum_threshold: Optional[float]
        :return: the proper revision operator evaluator
        :rtype: RevisionOperatorEvaluator
        """
        pass


class SingleRevisionOperatorEvaluator(RevisionOperatorEvaluatorSelector):
    """
    Selects the only operator.
    """

    def __init__(self, operator_evaluators):
        """
        Create a single revision operator selector.

        :param operator_evaluators: the operator evaluators
        :type operator_evaluators: Collection[RevisionOperatorEvaluator]
        """
        self.operator_evaluator = next(iter(operator_evaluators))

    # noinspection PyMissingOrEmptyDocstring
    def select_operator(self, targets, metric, minimum_threshold=None):
        if self.operator_evaluator is not None:
            self.operator_evaluator.clear_cached_theory()
        return self.operator_evaluator


class BestSelector(RevisionOperatorEvaluatorSelector):
    """
    Selects the best possible operator.
    """

    def __init__(self, operator_evaluators):
        """
        Create a best operator selector.

        :param operator_evaluators: the operator evaluators
        :type operator_evaluators: Collection[RevisionOperatorEvaluator]
        """
        self.operator_evaluators = operator_evaluators
        self.preferred_operator = next(iter(operator_evaluators))

    # noinspection PyMissingOrEmptyDocstring
    def select_operator(self, targets, metric, minimum_threshold=None):
        best_evaluator = self.preferred_operator
        best_evaluation = metric.default_value

        for evaluator in self.operator_evaluators:
            try:
                evaluator.clear_cached_theory()
                current = evaluator.evaluate_operator(
                    targets, metric, minimum_threshold)
                if metric.compare(current, best_evaluation) > 0:
                    best_evaluation = current
                    best_evaluator = evaluator
            except TheoryRevisionException:
                logger.exception(
                    "Error when evaluating the revision operator, reason:")

        return best_evaluator
