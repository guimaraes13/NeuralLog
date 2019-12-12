"""
File to define custom functions to use in the network.
"""

import tensorflow as tf
import tensorflow.keras
from tensorflow_core.python import keras

import src.network.layer_factory
from src.language.language import Predicate
from src.network import registry

TENSOR_FLOAT32_MAX = tf.constant(tf.float32.max)

CLIP_VALUE_MIN = tf.constant(1.0)

SPARSE_FUNCTION_SUFFIX = ":sparse"

literal_functions = dict()
combining_functions = dict()
initializers = dict()


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


@neural_log_combining_function("unary_literal_extraction_function")
def unary_literal_extraction_function(a, b):
    """
    Returns the function to extract the value of unary prediction.

    The default is the dot multiplication, implemented by the `tf.matmul`,
    applied to the transpose of the literal prediction.

    :param a: the first input tensor
    :type a: tf.Tensor
    :param b: the second input tensor
    :type b: tf.Tensor
    :return: the result tensor
    :rtype: tf.Tensor
    """
    result = tf.math.multiply(a, b)
    return tf.reduce_sum(result, axis=1)


@neural_log_literal_function("mean")
def mean(a):
    """
    Returns the mean of the non-zero values of `a` for each row.

    :param a: the input tensor
    :type a: tf.Tensor
    :return: the result tensor
    :rtype: tf.Tensor
    """
    sum_a = tf.math.reduce_sum(a, axis=1, keepdims=False)
    non_zero = tf.math.count_nonzero(a, axis=1,
                                     keepdims=False, dtype=tf.float32)
    non_zero = tf.clip_by_value(non_zero, CLIP_VALUE_MIN, TENSOR_FLOAT32_MAX)
    return tf.reshape(sum_a / non_zero, [-1, 1])


@neural_log_literal_function("square_root")
def square_root(a):
    """
    Returns the square root of the values of `a`.

    :param a: the input tensor
    :type a: tf.Tensor
    :return: the result tensor
    :rtype: tf.Tensor
    """
    return tf.math.sqrt(a)


@neural_log_literal_function("inverse")
def inverse(a):
    """
    Returns the inverse of the values of `a`.

    :param a: the input tensor
    :type a: tf.Tensor
    :return: the result tensor
    :rtype: tf.Tensor
    """
    return tf.math.divide_no_nan(1.0, a)


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
        self.kernel = kernel
        self.fact_combining_function = fact_combining_function
        self.rank = self.kernel.shape.rank
        super(FactLayer, self).__init__(name, **kwargs)

    def get_kernel(self):
        """
        Gets the processed kernel to apply the fact combining function.

        :return: the kernel
        :rtype: tf.Tensor
        """
        return self.kernel

    # # noinspection PyMissingOrEmptyDocstring
    # def key(self):
    #     return self.kernel, self.fact_combining_function

    def __hash__(self):
        return hash(self.fact_combining_function)

    def __eq__(self, other):
        # if isinstance(other, LiteralLayer):
        #     if other.negation_function is not None:
        #         return False
        #     if len(other.input_layers) != 1:
        #         return False
        #     return self == other.input_layers[0]
        #
        # if isinstance(other, RuleLayer):
        #     if len(other.paths) != 1:
        #         return False
        #     if len(other.grounded_layers) != 0:
        #         return False
        if not isinstance(other, FactLayer):
            return False

        return self.kernel is other.kernel and \
               self.fact_combining_function == other.fact_combining_function

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        if self.rank == 2 and inputs.shape.rank == 0:
            inputs = tf.fill([1, self.get_kernel().shape[0]], inputs)
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
        self.kernel = tf.linalg.tensor_diag_part(
            super(DiagonalFactLayer, self).get_kernel())


class InvertedFactLayer(FactLayer):
    """
    Represents a inverted fact layer.
    """

    def __init__(self, fact_layer, layer_factory, predicate, **kwargs):
        """
        Creates a InvertedFactLayer.

        :param fact_layer: the fact layer
        :type fact_layer: FactLayer
        :param layer_factory: the layer factory
        :type inverted_function: src.network.layer_factory.LayerFactory
        :param predicate: the atom's predicate
        :type predicate: Predicate
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        name = fact_layer.name + "_inv"
        self.kernel = fact_layer.get_kernel()
        self.fact_combining_function = fact_layer.fact_combining_function
        self._adjust_for_inverted_fact(layer_factory, predicate)
        super(InvertedFactLayer, self).__init__(
            name, self.kernel, self.fact_combining_function, **kwargs)

    def _adjust_for_inverted_fact(self, factory, predicate):
        """
        Adjusts the kernel and combining function to address the inverse
        predicate with constants in it.

        :param factory: the layer factory
        :type factory: src.network.layer_factory.LayerFactory
        :param predicate: the atom's predicate
        :type predicate: Predicate
        """
        sparse = isinstance(self.kernel, tf.SparseTensor)
        if self.kernel.shape.rank == 1:
            self.fact_combining_function = \
                factory.get_edge_combining_function_2d(predicate, sparse)
            if sparse:
                self.kernel = tf.sparse.reshape(self.kernel, [1, -1])
            else:
                self.kernel = tf.reshape(self.kernel, [1, -1])

        inverted_func = factory.get_invert_fact_function(predicate, sparse)
        self.kernel = inverted_func(self.kernel)  # type: tf.Tensor

        if self.kernel.shape.rank == 2 and self.kernel.shape[0] == 1:
            if sparse:
                self.kernel = tf.sparse.to_dense(self.kernel)
            self.fact_combining_function = \
                factory.get_edge_combining_function(predicate)
            self.kernel = tf.reshape(self.kernel, [-1])


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
        kernel = weight_combining_function(weights, values)
        super(AttributeFactLayer, self).__init__(
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
        self.fact_layer = fact_layer
        self.input_constant = input_constant
        self.inputs_combining_function = input_combining_function
        self.output_constant = output_constant
        self.output_extract_function = output_extract_function
        super(SpecificFactLayer, self).__init__(name, **kwargs)

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
            result = tf.reshape(result, [-1, 1])
        return result

    def __hash__(self):
        return hash((self.fact_layer,
                     self.inputs_combining_function,
                     self.output_extract_function))

    # noinspection DuplicatedCode
    def __eq__(self, other):
        if not isinstance(other, SpecificFactLayer):
            return False

        if self.fact_layer != other.fact_layer:
            return False

        if self.input_constant is not other.input_constant:
            return False

        if self.output_constant is not other.output_constant:
            return False

        if self.inputs_combining_function != other.inputs_combining_function:
            return False

        if self.output_extract_function != other.output_extract_function:
            return False

        return True


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
        self.fact_layer = fact_layer
        self.fact_combining_function = fact_combining_function
        self.output_constant = output_constant
        self.output_extract_function = output_extract_function
        self.kernel = tf.reshape(self.fact_layer(self.output_constant), [-1, 1])
        super(InvertedSpecificFactLayer, self).__init__(name, **kwargs)

    # noinspection PyMissingOrEmptyDocstring
    def get_kernel(self):
        return self.kernel

    # # noinspection PyMissingOrEmptyDocstring
    # def key(self):
    #     # noinspection PyTypeChecker
    #     return self.fact_layer.key() + (self.kernel, self.output_constant,
    #                                     self.fact_combining_function,
    #                                     self.output_extract_function)

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        # return self.fact_combining_function(self.get_kernel(), inputs)
        return self.fact_combining_function(inputs, self.get_kernel())

    def __hash__(self):
        return hash((self.fact_layer,
                     self.fact_combining_function,
                     self.output_extract_function))

    # noinspection DuplicatedCode
    def __eq__(self, other):
        if not isinstance(other, InvertedSpecificFactLayer):
            return False

        if self.fact_layer != other.fact_layer:
            return False

        if self.kernel is not other.kernel:
            return False

        if self.output_constant is not other.output_constant:
            return False

        if self.fact_combining_function != other.fact_combining_function:
            return False

        if self.output_extract_function != other.output_extract_function:
            return False

        return True


class LiteralLayer(NeuralLogLayer):
    """
    A Layer to combine the inputs of a literal. The inputs of a literal are
    the facts of the literal and the result of rules with the literal in
    their heads.
    """

    def __init__(self, name, input_layers, literal_combining_function,
                 negation_function=None, **kwargs):
        """
        Creates a LiteralLayer.

        :param name: the name of the layer
        :type name: str
        :param input_layers: the input layers.
        :type input_layers: List[FactLayer or RuleLayer]
        :param literal_combining_function: the literal combining function
        :type literal_combining_function: function
        :param negation_function: the literal negation function
        :type negation_function: function or None
        :param kwargs: additional arguments
        :type kwargs: dict
        """
        self.input_layers = input_layers
        self.literal_combining_function = literal_combining_function
        self.negation_function = negation_function
        super(LiteralLayer, self).__init__(name, **kwargs)

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        if len(self.input_layers) == 1:
            result = self.input_layers[0](inputs)
            if self.negation_function is not None:
                return self.negation_function(result)
            return result

        result = self.input_layers[0](inputs)
        for input_layer in self.input_layers[1:]:
            layer_result = input_layer(inputs)
            result = self.literal_combining_function(result, layer_result)
        if self.negation_function is not None:
            return self.negation_function(result)
        return result

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(LiteralLayer, self).get_config()

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __hash__(self):
        if self.negation_function is None and len(self.input_layers) == 1:
            return hash(self.input_layers[0].fact_combining_function)

        return hash(tuple(self.input_layers) +
                    (self.literal_combining_function, self.negation_function))

    def __eq__(self, other):
        # if isinstance(other, FactLayer):
        #     if self.negation_function is not None:
        #         return False
        #     if len(self.input_layers) != 1:
        #         return False
        #     return self.input_layers[0] == other
        if not isinstance(other, LiteralLayer):
            return False

        if self.input_layers != other.input_layers:
            return False

        if self.literal_combining_function != other.literal_combining_function:
            return False

        if self.negation_function != other.negation_function:
            return False

        return True


class FunctionLayer(NeuralLogLayer):
    """
    A Layer to apply the function literal at the input.
    """

    def __init__(self, name, function, inputs=None, **kwargs):
        """
        Creates a FunctionLayer.

        :param name: the name of the layer
        :type name: str
        :param function: the function
        :type function: function
        """
        self.function = function
        self.inputs = inputs
        super(FunctionLayer, self).__init__(name, **kwargs)

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        if self.inputs is None:
            return self.function(inputs)

        return self.function(self.inputs)

    def __hash__(self):
        return hash(self.function)

    def __eq__(self, other):
        if not isinstance(other, FunctionLayer):
            return False

        if self.inputs is not other.inputs:
            return False

        if self.function != other.function:
            return False

        return True

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(FunctionLayer, self).get_config()

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AnyLiteralLayer(NeuralLogLayer):
    """
    Layer to represent the special `any` literal.
    """

    def __init__(self, name, aggregation_function, **kwargs):
        """
        Creates an AnyLiteralLayer

        :param name: the name of the layer
        :type name: str
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        self.aggregation_function = aggregation_function
        super(AnyLiteralLayer, self).__init__(name, **kwargs)

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        result = self.aggregation_function(inputs)
        result = tf.reshape(result, [-1, 1])
        return result

    def __hash__(self):
        return hash(self.aggregation_function)

    def __eq__(self, other):
        if not isinstance(other, AnyLiteralLayer):
            return False

        return self.aggregation_function == other.aggregation_function

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(AnyLiteralLayer, self).get_config()

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class RuleLayer(NeuralLogLayer):
    """
    A Layer to represent a logic rule.
    """

    def __init__(self, name, paths, grounded_layers, path_combining_function,
                 neutral_element, **kwargs):
        """
        Creates a RuleLayer.

        :param name: the name of the layer
        :type name: str
        :param paths: the paths of the layer
        :type paths: List[List[LiteralLayer]]
        :param grounded_layers: the grounded literal layers
        :type grounded_layers: List[LiteralLayer]
        :param path_combining_function: the path combining function
        :type path_combining_function: function
        :param neutral_element: the neural element to be passed to the
        grounded layer
        :type neutral_element: tf.Tensor
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        self.paths = paths
        self.grounded_layers = grounded_layers
        self.path_combining_function = path_combining_function
        self.neutral_element = neutral_element
        super(RuleLayer, self).__init__(name, **kwargs)

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        if len(self.paths) > 0:
            path_result = self._compute_path_tensor(inputs, self.paths[0])
            for i in range(1, len(self.paths)):
                tensor = self._compute_path_tensor(inputs, self.paths[i])
                path_result = self.path_combining_function(path_result, tensor)
        else:
            path_result = self.neutral_element
        for grounded_layer in self.grounded_layers:
            grounded_result = grounded_layer(self.neutral_element)
            path_result = self.path_combining_function(path_result,
                                                       grounded_result)
        return path_result

    def _compute_path_tensor(self, inputs, path):
        """
        Computes the path for the `inputs`.

        :param inputs: the inputs
        :type inputs: tf.Tensor
        :param path: the path
        :type path: List[LiteralLayer]
        :return: the computed path
        :rtype: tf.Tensor
        """
        tensor = inputs
        for i in range(len(path)):
            literal_layer = path[i]
            if isinstance(literal_layer, AnyLiteralLayer):
                new_tensor = self._compute_path_tensor(self.neutral_element,
                                                       path[i + 1:])
                new_tensor = tf.reshape(new_tensor, [1, -1])
                tensor = literal_layer(tensor)
                return tf.matmul(tensor, new_tensor)
            tensor = literal_layer(tensor)
        return tensor

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(RuleLayer, self).get_config()

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __hash__(self):
        return hash((len(self.paths), self.path_combining_function) +
                    tuple(self.grounded_layers))

    def __eq__(self, other):
        if not isinstance(other, RuleLayer):
            return False

        if self.path_combining_function != other.path_combining_function:
            return False

        if self.paths != other.paths:
            return False

        if self.grounded_layers != other.grounded_layers:
            return False

        return True


class ExtractUnaryLiteralLayer(NeuralLogLayer):
    """
    Class to extract the value of a unary literal predicate prediction.
    """

    def __init__(self, literal_layer, input_combining_function, **kwargs):
        """
        Creates a SpecificRuleLayer.

        :param literal_layer: the more unary rule layer
        :type literal_layer: LiteralLayer
        :type input_combining_function: function
        :param input_combining_function: the function to combine input with
        the output of the literal layer
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        name = literal_layer.name + "_extract_unary"
        self.literal_layer = literal_layer
        self.input_combining_function = input_combining_function
        super(ExtractUnaryLiteralLayer, self).__init__(name, **kwargs)

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        result = self.literal_layer(inputs)
        return self.input_combining_function(inputs, result)

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = 1
        return tuple(output_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(ExtractUnaryLiteralLayer, self).get_config()

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __hash__(self):
        return hash(self.literal_layer)

    # noinspection DuplicatedCode
    def __eq__(self, other):
        if not isinstance(other, ExtractUnaryLiteralLayer):
            return False

        if self.literal_layer != other.literal_layer:
            return False

        if self.input_combining_function is not other.input_combining_function:
            return False

        return True
