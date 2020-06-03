"""
Handles the communication between the structure learning algorithm and the
inference engine.
"""
from abc import ABC

from src.util import Initializable


class EngineSystemTranslator(ABC, Initializable):
    """
    Translates the results of the engine system to the structure learning
    algorithm and vice versa.
    """

    def __init__(self, program):
        self.program = program
        self.model = None

    def initialize(self):
        pass
        # self.model = NeuralLogNetwork()
