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
    def infer_examples(self, examples):
        """
        Perform the inference for the given examples.

        :param examples: the examples
        :type examples: Examples
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
