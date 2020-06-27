"""
Package with generic useful tools.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Iterator, Iterable, Union, Tuple, AbstractSet, Set, \
    TypeVar

from src.run.command import SKIP_SIZE, INDENT_SIZE

logger = logging.getLogger(__name__)


# noinspection PyMissingOrEmptyDocstring
class OrderedSet(set):
    """
    An set that iterates in the same order in the elements were added.
    """
    _T = TypeVar('_T')
    _S = TypeVar('_S')

    def __init__(self, *iterable: Iterable[_T]) -> None:
        super().__init__()
        self._list = []
        self.update(*iterable)

    def add(self, element: _T) -> None:
        if element not in self:
            super().add(element)
            self._list.append(element)

    def clear(self) -> None:
        super().clear()
        self._list.clear()

    def copy(self) -> Set[_T]:
        return OrderedSet(self)

    def difference(self, *s: Iterable[Any]) -> Set[_T]:
        return super().difference(*s)

    def difference_update(self, *s: Iterable[Any]) -> None:
        super().difference_update(*s)

    def discard(self, element: _T) -> None:
        if element in self:
            super().discard(element)
            self._list.remove(element)

    def intersection(self, *s: Iterable[Any]) -> Set[_T]:
        return super().intersection(*s)

    def intersection_update(self, *s: Iterable[Any]) -> None:
        self.update(self.intersection(*s))

    def isdisjoint(self, s: Iterable[Any]) -> bool:
        return super().isdisjoint(s)

    def issubset(self, s: Iterable[Any]) -> bool:
        return super().issubset(s)

    def issuperset(self, s: Iterable[Any]) -> bool:
        return super().issuperset(s)

    def pop(self) -> _T:
        value = self._list[-1]
        self.remove(value)
        return value

    def remove(self, element: _T) -> None:
        super().remove(element)
        self._list.remove(element)

    def symmetric_difference(self, s: Iterable[_T]) -> Set[_T]:
        return super().symmetric_difference(s)

    def symmetric_difference_update(self, s: Iterable[_T]) -> None:
        self.update(self.symmetric_difference(s))

    def union(self, *s: Iterable[_T]) -> Set[_T]:
        return super().union(*s)

    def update(self, *s: Iterable[_T]) -> None:
        for sequence in s:
            for item in sequence:
                self.add(item)

    def __len__(self) -> int:
        return super().__len__()

    def __contains__(self, o: object) -> bool:
        return super().__contains__(o)

    def __iter__(self) -> Iterator[_T]:
        return iter(self._list)

    def __str__(self) -> str:
        return self.__repr__()

    def __and__(self, s: AbstractSet[object]) -> Set[_T]:
        return super().__and__(s)

    def __iand__(self, s: AbstractSet[object]) -> Set[_T]:
        return super().__iand__(s)

    def __or__(self, s: AbstractSet[_S]) -> Set[Union[_T, _S]]:
        return super().__or__(s)

    def __ior__(self, s: AbstractSet[_S]) -> Set[Union[_T, _S]]:
        return super().__ior__(s)

    def __sub__(self, s: AbstractSet[object]) -> Set[_T]:
        return super().__sub__(s)

    def __isub__(self, s: AbstractSet[object]) -> Set[_T]:
        return super().__isub__(s)

    def __xor__(self, s: AbstractSet[_S]) -> Set[Union[_T, _S]]:
        return super().__xor__(s)

    def __ixor__(self, s: AbstractSet[_S]) -> Set[Union[_T, _S]]:
        return super().__ixor__(s)

    def __le__(self, s: AbstractSet[object]) -> bool:
        return super().__le__(s)

    def __lt__(self, s: AbstractSet[object]) -> bool:
        return super().__lt__(s)

    def __ge__(self, s: AbstractSet[object]) -> bool:
        return super().__ge__(s)

    def __gt__(self, s: AbstractSet[object]) -> bool:
        return super().__gt__(s)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o)

    def __ne__(self, o: object) -> bool:
        return super().__ne__(o)

    def __repr__(self) -> str:
        return "{{{}}}".format(", ".join(map(lambda x: str(x), self._list)))

    def __hash__(self) -> int:
        return sum(map(lambda x: hash(x), self._list))

    def __format__(self, format_spec: str) -> str:
        return super().__format__(format_spec)

    def __getattribute__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)

    def __sizeof__(self) -> int:
        return super().__sizeof__()

    def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
        return super().__reduce__()

    def __reduce_ex__(self, protocol: int) -> Union[str, Tuple[Any, ...]]:
        return super().__reduce_ex__(protocol)

    def __dir__(self) -> Iterable[str]:
        return super().__dir__()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()


class InitializationException(Exception):
    """
    Represents an initialization exception.
    """

    def __init__(self, message):
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

    OPTIONAL_FIELDS = {}

    def initialize(self):
        """
        Initializes the object.

        :raise InitializationException: if an error occurs during the
        initialization of the object
        """
        logger.debug("Initializing:\t%s", self.__class__.__name__)
        self._initialize_fields()
        fields = []
        required_fields = self.required_fields()
        if required_fields is None:
            return

        for field in required_fields:
            if not hasattr(self, field) or getattr(self, field) is None:
                fields.append(field)

        if len(fields) > 0:
            raise unset_fields_error(fields, self)

    def _initialize_fields(self):
        """
        Initialize the fields. Since the `__init__` method of a class is not
        called when it is constructed by the pyyaml library, this method
        gives an opportunity to initiate the fields of the classes, mainly
        the ones that are optional and have default values.
        """
        for key, value in self.OPTIONAL_FIELDS.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    @abstractmethod
    def required_fields(self):
        """
        Returns a list of the required fields of the class.

        :return: the list of required fields
        :rtype: list[str]
        """
        return []

    def __setattr__(self, name: str, value: Any):
        required_fields = self.required_fields()
        if required_fields is not None and name in required_fields:
            if hasattr(self, name) and getattr(self, name) is not None:
                raise reset_field_error(self, "learning_system")
        super().__setattr__(name, value)

    def __repr__(self):
        return self.__class__.__name__


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


def print_args(args, the_logger):
    """
    Prints the parsed arguments in an organized way.

    :param args: the parsed arguments
    :type args: argparse.Namespace or dict
    :param the_logger: the logger
    :type the_logger: logger
    """
    if isinstance(args, dict):
        arguments = args
    else:
        arguments = args.__dict__
    max_key_length = max(
        map(lambda x: len(str(x)), arguments.keys()))
    for k, v in sorted(arguments.items(), key=lambda x: str(x)):
        if hasattr(v, "__len__") and len(v) == 1 and not isinstance(v, dict):
            v = v[0]
        k = str(k)
        the_logger.info("%s:%s%s", k,
                        " " * (int((max_key_length - len(k)) / SKIP_SIZE)
                               + INDENT_SIZE), v)
    the_logger.info("")
