"""
Handles the communication between the structure learning algorithm and the
inference engine.
"""
from abc import abstractmethod

from src.knowledge.examples import Examples, ExamplesInferences
from src.util import Initializable


class EngineSystemTranslator(Initializable):
    """
    Translates the results of the engine system to the structure learning
    algorithm and vice versa.
    """

    def __init__(self):
        self.program = None
        self.model = None

    @abstractmethod
    def infer_examples(self, examples, retrain=False, theory=None):
        """
        Perform the inference for the given examples. If `retrain` is `True`,
        it retrains the parameters before inference. If `theory` is not
        `None`, use it as the theory to be evaluated, instead of the current
        theory of the engine system.

        :param examples: the examples
        :type examples: Examples
        :param retrain: if `True`, it retrains the parameters before inference
        :type retrain: bool
        :param theory: If not `None`, use this theory instead of the current
        theory of the engine system.
        :type theory: NeuralLogProgram
        :return: the inference value of the examples
        :rtype: ExamplesInferences
        """
        pass

    @abstractmethod
    def infer_examples_appending_clauses(self, examples, clauses,
                                         retrain=False):
        """
        Perform the inference for the given examples. The `clauses` are
        appended to the current theory, before evaluation. If `retrain` is
        `True`, it retrains the parameters before inference.

        After evaluation, the appended clauses are discarded and the parameters
        are restored to its values previously to the evaluation.

        :param examples: the examples
        :type examples: Examples
        :param retrain: if `True`, it retrains the parameters before inference
        :type retrain: bool
        :param clauses: clauses to be appended to the current theory, for
        evaluation proposes only
        :type clauses: collections.Iterable[HornClauses]
        :return: the inference value of the examples
        :rtype: ExamplesInferences
        """
        pass

    @abstractmethod
    def train_parameters(self, training_examples):
        """
        Trains the parameters of the model.

        :param training_examples: the training examples
        :type training_examples: Examples
        """
        pass

    @abstractmethod
    def save_trained_parameters(self):
        """
        Saves the trained parameters.
        """
        pass
