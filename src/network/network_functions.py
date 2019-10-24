"""
File to define custom functions to use in the network.
"""

import tensorflow as tf
import tensorflow.keras

literal_functions = dict()
combining_functions = dict()
initializers = dict()


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


def neural_log_literal_function(identifier):
    """
    A decorator for NeuralLog literal functions.

    :param identifier: the identifier of the function
    :type identifier: str
    :return: the decorated function
    :rtype: function
    """
    return lambda x: registry(x, identifier, literal_functions)


def neural_log_combining_function(identifier):
    """
    A decorator for NeuralLog combining functions.

    :param identifier: the identifier of the function
    :type identifier: str
    :return: the decorated function
    :rtype: function
    """
    return lambda x: registry(x, identifier, combining_functions)


def neural_log_initializer(identifier):
    """
    A decorator for NeuralLog variable initializers.

    :param identifier: the identifier of the function
    :type identifier: str
    :return: the decorated function
    :rtype: function
    """
    return lambda x: registry(x, identifier, initializers)


# noinspection PyMissingOrEmptyDocstring
@neural_log_literal_function("my_print_value")
def print_value(value1, value2):
    print(value1, value2)


# noinspection PyMissingOrEmptyDocstring,PyShadowingBuiltins
@neural_log_literal_function("my_func")
class MyFunc:

    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max

    def __call__(self, *args, **kwargs):
        print("Class min:\t{}\nClass max:\t{}\nArguments:\t{}".format(
            self.min, self.max, args))


# @neural_log_literal_function("my_const_value")
# noinspection PyMissingOrEmptyDocstring
@neural_log_initializer("my_const_value")
class ConstantLiteral:

    def __init__(self, value):
        self.value = value

    def __call__(self, *args, **kwargs):
        return tf.constant(self.value, shape=args[0])


# @neural_log_literal_function("my_const_value_2")
# noinspection PyMissingOrEmptyDocstring,PyUnusedLocal
@neural_log_initializer("my_const_value_2")
def constant_literal_2(*args, **kwargs):
    return tf.constant(2.0, shape=args[0])


def _deserialize(configuration, function_dict, keras_func, name_only=False):
    """
    Gets the function object from `configuration`.

    :param configuration: the configuration
    :type configuration: str or dict[str, str or dict]
    :param function_dict: a dictionary with the function objects
    :type function_dict: dict[str, function]
    :param keras_func: the keras function to get the function from, if it is
    not in `function_dict`
    :type keras_func: function or None
    :raise ValueError: if the function is not found
    :return: the function
    :rtype: function
    """
    class_name = configuration["class_name"]
    found_object = function_dict.get(class_name, None)
    if found_object is None:
        if keras_func is None:
            raise ValueError('Could not interpret initializer identifier: ' +
                             class_name)
        if name_only:
            return keras_func(class_name)
        else:
            return keras_func(configuration)
    elif isinstance(found_object, type):
        return found_object(**configuration["config"])
    else:
        return found_object


def _get(identifier, function_dict, keras_func, name_only=False):
    """
    Gets the function object from `identifier`.

    :param identifier: the identifier
    :type identifier: str or dict[str, str or dict]
    :param function_dict: a dictionary with the function objects
    :type function_dict: dict[str, function]
    :param keras_func: the keras function to get the function from, if it is
    not in `function_dict`
    :type keras_func: function or None
    :raise ValueError: if the function is not found
    :return: the function
    :rtype: function
    """
    if identifier is None:
        return None
    if isinstance(identifier, str):
        identifier = {"class_name": identifier, "config": {}}

    if isinstance(identifier, dict):
        return _deserialize(identifier, function_dict, keras_func, name_only)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret initializer identifier: ' +
                         str(identifier))


def get_initializer(identifier):
    """
    Gets the variable initializer from `identifier`.

    :param identifier: the identifier
    :type identifier: str or dict[str, str or dict]
    :raise ValueError: if the function is not found
    :return: the function
    :rtype: function
    """
    return _get(identifier, initializers, tensorflow.keras.initializers.get)


def get_literal_function(identifier):
    """
    Gets the literal function from `identifier`.

    :param identifier: the identifier
    :type identifier: str or dict[str, str or dict]
    :raise ValueError: if the function is not found
    :return: the function
    :rtype: function
    """
    return _get(identifier, literal_functions,
                tensorflow.keras.activations.get, name_only=True)


def get_combining_function(identifier):
    """
    Gets the combining function from `identifier`.

    :param identifier: the identifier
    :type identifier: str or dict[str, str or dict]
    :raise ValueError: if the function is not found
    :return: the function
    :rtype: function
    """
    if isinstance(identifier, str) and identifier.startswith("tf."):
        identifiers = identifier.split(".")[1:]
        func = None
        for name in identifiers:
            func = getattr(tf, name)
        return func
    else:
        return _get(identifier, combining_functions, None)


if __name__ == "__main__":
    get_literal_function("my_print_value")(2, 3)
    config = {"class_name": "my_func", "config": {"min": 5, "max": 10}}
    get_literal_function(config)(4, 5, 6)
    get_literal_function("my_func")(0.2, 0.3)
