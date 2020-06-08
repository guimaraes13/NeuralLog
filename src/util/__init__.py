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

        :param message: the message
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


def reset_field_error(field_name, clazz):
    """
    Error for resetting a field.

    :param field_name: the field name
    :type field_name: str
    :param clazz: the class
    :type clazz: class
    :return: the error
    :rtype: InitializationException
    """
    return InitializationException(
        "Reset {}, at class {}, is not allowed.".format(
            field_name, clazz.__class__.__name__))
