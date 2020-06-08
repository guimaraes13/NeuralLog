"""
Handles the communication between the structure learning algorithm and the
inference engine.
"""
from abc import ABC, abstractmethod

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
