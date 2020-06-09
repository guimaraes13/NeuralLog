"""
Handles the communication between the structure learning algorithm and the
inference engine.
"""
from abc import abstractmethod

from src.util import Initializable


class EngineSystemTranslator(Initializable):
    """
    Translates the results of the engine system to the structure learning
    algorithm and vice versa.
    """

    def __init__(self):
        self.program = None
        self.model = None

    # noinspection PyMissingOrEmptyDocstring
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def infer_examples(self, examples):
        """
        Perform the inference for the given examples.

        :param examples: the examples
        :type examples: Dict[Predicate, Dict[Any, Atom]]
        :return: the inference value of the examples
        :rtype: Dict[Predicate, Dict[Any, float]]
        """
        pass
