"""
Manages the incoming examples.
"""

from abc import ABC

from src.util import Initializable


class IncomingExampleManager(ABC, Initializable):
    """
    Responsible for receiving the examples.
    """

    def __init__(self, learning_system, sample_selector):
        self.learning_system = learning_system
        self.sample_selector = sample_selector

