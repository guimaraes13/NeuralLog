"""
File to define custom functions to use in the network.
"""

import tensorflow as tf
import tensorflow.keras
from tensorflow_core.python import keras

SPARSE_FUNCTION_SUFFIX = ":sparse"

literal_functions = dict()
combining_functions = dict()
initializers = dict()


class NeuralLogLayer(keras.layers.Layer):
    """
    Represents a NeuralLogLayer.
    """

    def __init__(self, name, **kwargs):
        # noinspection PyTypeChecker
        kwargs["name"] = name
        self.layer_name = name
        super(NeuralLogLayer, self).__init__(**kwargs)

    def __str__(self):
        return "[{}] {}".format(self.__class__.__name__, self.layer_name)

    __repr__ = __str__


class AbstractFactLayer(NeuralLogLayer):
    """
    Represents an abstract fact layer.
    """

    def __init__(self, name, **kwargs):
        """
        Creates an AbstractFactLayer.
        :param name: the name of the layer
        :type name: str
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        super(AbstractFactLayer, self).__init__(name, **kwargs)

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(AbstractFactLayer, self).get_config()

    # noinspection PyMissingOrEmptyDocstring,PyShadowingNames
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FactLayer(AbstractFactLayer):
    """
    Represents a simple fact layer.
    """

    def __init__(self, name, kernel, fact_combining_function, **kwargs):
        """
        Creates a SimpleFactLayer.

        :param name: the name of the layer
        :type name: str
        :param kernel: the data of the layer.
        :type kernel: tf.Tensor
        :param fact_combining_function: the fact combining function
        :type fact_combining_function: function
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        super(FactLayer, self).__init__(name, **kwargs)
        self.kernel = kernel
        self.fact_combining_function = fact_combining_function

    def get_kernel(self):
        """
        Gets the processed kernel to apply the fact combining function.

        :return: the kernel
        :rtype: tf.Tensor
        """
        return self.kernel

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        return self.fact_combining_function(inputs, self.get_kernel())


class DiagonalFactLayer(FactLayer):
    """
    Represents a simple fact layer.
    """

    def __init__(self, name, kernel, fact_combining_function, **kwargs):
        """
        Creates a SimpleFactLayer.

        :param name: the name of the layer
        :type name: str
        :param kernel: the data of the layer.
        :type kernel: tf.Tensor
        :param fact_combining_function: the fact combining function
        :type fact_combining_function: function
        :param kernel: the data of the layer.
        :type kernel: tf.Tensor
        :param fact_combining_function: the fact combining function
        :type fact_combining_function: function
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        super(DiagonalFactLayer, self).__init__(
            name, kernel, fact_combining_function, **kwargs)

    # noinspection PyMissingOrEmptyDocstring
    def get_kernel(self):
        return tf.linalg.tensor_diag_part(
            super(DiagonalFactLayer, self).get_kernel())


class InvertedFactLayer(FactLayer):
    """
    Represents a inverted fact layer.
    """

    def __init__(self, fact_layer, inverted_function, **kwargs):
        """
        Creates a InvertedFactLayer.

        :param fact_layer: the fact layer
        :type fact_layer: FactLayer
        :param inverted_function: the fact inversion function. The function to
        extract the inverse of the facts
        :type inverted_function: function
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        name = fact_layer.name + "_inv"
        kernel = inverted_function(fact_layer.get_kernel())
        fact_combining_function = fact_layer.fact_combining_function
        super(InvertedFactLayer, self).__init__(
            name, kernel, fact_combining_function, **kwargs)


class SpecificFactLayer(AbstractFactLayer):
    """
    A layer to represent a fact with constants applied to it.
    """

    def __init__(self, name, fact_layer,
                 input_constant=None,
                 input_combining_function=None, output_constant=None,
                 output_extract_function=None, **kwargs):
        """
        Creates a PredicateLayer.

        :param fact_layer: a fact layer
        :type fact_layer: AbstractFactLayer
        :param fact_combining_function: the fact combining function
        :type fact_combining_function: function
        :param input_constant: the input constant
        :type input_constant: tf.Tensor
        :param input_combining_function: the function to combine the fixed
        input with the input of the layer
        :type input_combining_function: function
        :param output_constant: the output constant, if any
        :type output_constant: tf.Tensor
        :param output_extract_function: the function to extract the fact value
        of the fixed output constant
        :type output_extract_function: function
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        super(SpecificFactLayer, self).__init__(name, **kwargs)
        self.fact_layer = fact_layer
        self.input_constant = input_constant
        self.inputs_combining_function = input_combining_function
        self.output_constant = output_constant
        self.output_extract_function = output_extract_function

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        if self.input_constant is None:
            input_constant = inputs
        else:
            input_constant = self.inputs_combining_function(self.input_constant,
                                                            inputs)
        result = self.fact_layer.call(input_constant)
        if self.output_constant is not None:
            if len(result.shape.as_list()) == 2:
                result = tf.transpose(result)
            result = self.output_extract_function(result, self.output_constant)
        return result


class InvertedSpecificFactLayer(AbstractFactLayer):
    """
    A layer to represent a inverted fact with constants applied to it.
    """

    def __init__(self, name, fact_layer, fact_combining_function,
                 output_constant=None, output_extract_function=None, **kwargs):
        """
        Creates a PredicateLayer.

        :param fact_layer: a fact layer
        :type fact_layer: AbstractFactLayer
        :param fact_combining_function: the fact combining function
        :type fact_combining_function: function
        :param output_constant: the output constant, if any
        :type output_constant: tf.Tensor
        :param output_extract_function: the function to extract the fact value
        of the fixed output constant
        :type output_extract_function: function
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        super(InvertedSpecificFactLayer, self).__init__(name, **kwargs)
        self.fact_layer = fact_layer
        self.fact_combining_function = fact_combining_function
        self.output_constant = output_constant
        self.output_extract_function = output_extract_function

    # noinspection PyMissingOrEmptyDocstring
    def get_kernel(self):
        kernel = self.fact_layer.get_kernel()
        return self.output_extract_function(kernel, self.output_constant)

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        return self.fact_combining_function(self.get_kernel(), inputs)


class AttributeFactLayer(FactLayer):
    """
    Represents an attribute fact layer.
    """

    def __init__(self, name, weights, values,
                 fact_combining_function, weight_combining_function, **kwargs):
        """
        Creates an AttributeFactLayer.

        :param name: the name of the layer
        :type name: str
        :param weights: the weights of the layer.
        :type weights: tf.Tensor
        :param weights: the values of the layer.
        :type weights: tf.Tensor
        :param fact_combining_function: the fact combining function
        :type fact_combining_function: function
        :param weight_combining_function: the function to combine the weights
        and values of the attribute facts
        :type weight_combining_function: function
        :param attributes_combining_function: the function to
        combine the numeric terms of a fact
        :type attributes_combining_function: function
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        super(AttributeFactLayer, self).__init__(
            name, weights, fact_combining_function, **kwargs)
        self.values = values
        self.weight_combining_function = weight_combining_function

    # noinspection PyMissingOrEmptyDocstring
    def get_kernel(self):
        return self.weight_combining_function(self.kernel, self.values)


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
    :rtype: function or tf.Tensor
    """
    configuration = None
    if isinstance(identifier, dict):
        class_name = identifier["class_name"]
        configuration = identifier["config"]
    else:
        class_name = identifier
    if class_name.startswith("tf."):
        names = class_name.split(".")[1:]
        func = tf
        for name in names:
            func = getattr(func, name)
        if configuration is None:
            return func
        else:
            return func(**configuration)
    else:
        return _get(identifier, combining_functions, None)


@neural_log_combining_function("edge_combining_function_2d:sparse")
def edge_combining_function_2d_sparse(a, sp_b):
    """
    Combines the tensors `a` and `sp_b`, where `sp_b` is a sparse tensor.

    :param a: the tensor a
    :type a: tf.Tensor
    :param sp_b: the tensor b
    :type sp_b: tf.SparseTensor
    :return: a combination of the tensor
    :rtype: tf.Tensor
    """
    tensor = tf.sparse.sparse_dense_matmul(sp_b, a,
                                           adjoint_a=True, adjoint_b=True)
    return tf.transpose(tensor)


@neural_log_literal_function("literal_negation_function")
def literal_negation_function(a):
    """
    Returns the negation of the atom tensor `a`.

    :param a: the atom tensor
    :type a: tf.Tensor
    :return: the negation of `a`
    :rtype: tf.Tensor
    """
    return tf.add(1.0, -a)


@neural_log_literal_function("literal_negation_function:sparse")
def literal_negation_function_sparse(a):
    """
    Returns the negation of the atom sparse tensor `a`.

    :param a: the atom sparse tensor
    :type a: tf.SparseTensor
    :return: the negation of `a`
    :rtype: tf.SparseTensor
    """
    return literal_negation_function(tf.sparse.to_dense(a))


# noinspection PyMissingOrEmptyDocstring


@neural_log_initializer("my_const_value")
class MyConstant:

    def __init__(self, value):
        self.value = value

    def __call__(self, shape):
        return tf.Variable(self.value, shape=shape)
        # return self.value


# noinspection PyMissingOrEmptyDocstring


@neural_log_initializer("my_const_value_2")
def my_const_value_2(shape):
    return tf.Variable(2.0, shape=shape)
    # return 2.0


@neural_log_combining_function("any_aggregation_function")
def any_aggregation_function(a):
    """
    Returns the function to aggregate the input of an `Any` predicate.

    The default function is the `tf.reduce_sum`.

    :param a: the input tensor
    :type a: tf.Tensor
    :return: the result tensor
    :rtype: tf.Tensor
    """
    return tf.reduce_sum(a, axis=1)


if __name__ == "__main__":
    get_literal_function("my_print_value")(2, 3)
    config = {"class_name": "my_func", "config": {"min": 5, "max": 10}}
    get_literal_function(config)(4, 5, 6)
    get_literal_function("my_func")(0.2, 0.3)
