"""
Package with generic useful tools.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class InitializationException(Exception):
    """
    Represents an initialization exception.
    """

    def __init__(self, message) -> None:
        """
        Creates an exception with message.

        :param message: the message
        :type message: str
        """
        super().__init__(message)


class Initializable(ABC):
    """
    Interface to allow object to be initialized.
    """

    def initialize(self) -> None:
        """
        Initializes the object.

        :raise InitializationException: if an error occurs during the
        initialization of the object
        """
        logger.debug("Initializing\t%s", self.__class__.__name__)
        fields = []
        required_fields = self.required_fields()
        if required_fields is None:
            return

        for field in required_fields:
            if not hasattr(self, field) or getattr(self, field) is None:
                fields.append("learning_system")

        if len(fields) > 0:
            raise unset_fields_error(fields, self)

    @abstractmethod
    def required_fields(self):
        """
        Returns a list of the required fields of the class.

        :return: the list of required fields
        :rtype: list[str]
        """
        pass

    def __setattr__(self, name: str, value: Any) -> None:
        required_fields = self.required_fields()
        if required_fields is not None and name in required_fields:
            if hasattr(self, name) and getattr(self, name) is not None:
                raise reset_field_error(self, "learning_system")
        super().__setattr__(name, value)


def unset_fields_error(field_name, clazz):
    """
    Error for unset field.

    :param field_name: the name of the unset field
    :type field_name: str or list[str]
    :param clazz: the class whose field was unset
    :type clazz: class
    :return: the error
    :rtype: InitializationException
    """
    suffix = ", at class {}, must be set prior initialization.".format(
        clazz.__class__.__name__)
    if not isinstance(field_name, list):
        message = "Field {}".format(field_name) + suffix
    elif len(field_name) == 1:
        message = "Field {}".format(field_name[0]) + suffix
    else:
        message = "Fields "
        message += ", ".join(field_name[:-1])
        message += " and "
        message += field_name[-1]
        message += suffix.format(clazz.__class__.__name__)

    return InitializationException(message)


def reset_field_error(clazz, field_name):
    """
    Error for resetting a field.

    :param clazz: the class
    :type clazz: class
    :param field_name: the field name
    :type field_name: str
    :return: the error
    :rtype: InitializationException
    """
    return InitializationException(
        "Reset {}, at class {}, is not allowed.".format(
            field_name, clazz.__class__.__name__))
