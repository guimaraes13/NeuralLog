"""
Package with generic useful tools.
"""
from abc import ABC, abstractmethod


class InitializationException(Exception):
    """
    Represents an initialization exception.
    """

    def __init__(self, message) -> None:
        """
        Creates an exception with message.

        :param message: the messate
        :type message: str
        """
        super().__init__(message)


class Initializable(ABC):
    """
    Interface to allow object to be initialized.
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initializes the object.

        :raise InitializationException: if an error occurs during the
        initialization of the object
        """
        pass
