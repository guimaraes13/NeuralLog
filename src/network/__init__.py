"""
Package to store the network modules.
"""
from typing import Callable


def registry(func: Callable, identifier, func_dict):
    """
    Registries the function or class.

    :param func: the function
    :type func: Callable
    :param identifier: the function identifier
    :type identifier: str
    :param func_dict: the dictionary to registry the function in
    :type func_dict: dict[str, Any]
    :return: the function
    :rtype: Callable
    """
    func_dict[identifier] = func
    return func
