"""
Package to store the network modules.
"""


def registry(func, identifier, func_dict):
    """
    Registries the function or class.

    :param func: the function
    :type func: function
    :param identifier: the function identifier
    :type identifier: str
    :param func_dict: the dictionary to registry the function in
    :type func_dict: dict[str, Any]
    :return: the function
    :rtype: function
    """
    func_dict[identifier] = func
    return func
