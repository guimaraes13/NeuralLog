"""
Handle the evaluation of the revision operators.
"""
from typing import Dict

from src.knowledge.program import NeuralLogProgram
from src.knowledge.theory.evaluation.metric.theory_metric import TheoryMetric
from src.knowledge.theory.manager.revision.operator.revision_operator import \
    RevisionOperator
from src.language.language import Predicate, Atom
from src.util import Initializable


# TODO: implement this class
class RevisionOperatorEvaluator(Initializable):
    """
    Class responsible for evaluating the revision operator.
    """

    def __init__(self, revision_operator: RevisionOperator):
        self.revision_operator = revision_operator

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["revision_operator"]

    def evaluate_operator(self, examples, theory_metric):
        """
        Evaluates the operator in the examples, based on the metric.

        :param examples: the examples
        :type examples: Dict[Predicate, Dict[Any, Atom]]
        :param theory_metric: the metric
        :type theory_metric: TheoryMetric
        :return: the evaluation of the operator
        :rtype: float
        """
        pass

    def get_revised_theory(self, training_examples):
        """
        Gets the revised theory. This method is useful because most of the 
        revision operators need to previously apply the change before 
        evaluating the theory. This method allows it to store the revised 
        theory, in order to improve performance.
        
        :param training_examples: the training examples
        :type training_examples: Dict[Predicate, Dict[Any, Atom]]
        :return: the revised theory
        :rtype: NeuralLogProgram
        """
        pass

    def theory_revision_accepted(self, revised_theory):
        """
        Method to send a feedback to the revision operator, telling that the
        revision was accepted.

        :param revised_theory: the revised theory
        :type revised_theory: NeuralLogProgram
        """
        self.revision_operator.theory_revision_accepted(revised_theory)
