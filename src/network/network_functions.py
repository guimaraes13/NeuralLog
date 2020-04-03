"""
File to define custom functions to use in the network.
"""
from collections import deque
from functools import reduce
from typing import Dict

import tensorflow as tf
import tensorflow.keras
from tensorflow.python.training.tracking import data_structures
from tensorflow_core.python import keras

import src.network.layer_factory
from src.knowledge.graph import RuleGraph, Edge
from src.knowledge.program import ANY_PREDICATE_NAME
from src.language.language import Predicate, Term, Literal, Atom
from src.network import registry

EMPTY_DICT = dict()

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


def get_literal_layer(identifier):
    """
    Gets the literal function from `identifier`.

    :param identifier: the identifier
    :type identifier: str or dict[str, str or dict]
    :raise ValueError: if the function is not found
    :return: the function
    :rtype: function
    """
    return _get(identifier, EMPTY_DICT, tensorflow.keras.layers.deserialize)


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


@neural_log_combining_function("concat_combining_function")
def concat_combining_function(a, b):
    """
    Combines the tensor `a` and `b` by concatenating them.

    :param a: the tensor a
    :type a: tf.Tensor
    :param b: the tensor b
    :type b: tf.Tensor
    :return: a combination of the tensors
    :rtype: tf.Tensor
    """
    if a.shape.rank == 2:
        a = tf.expand_dims(a, 1)
    if b.shape.rank == 2:
        b = tf.expand_dims(b, 1)
    return tf.concat([a, b], axis=1)


@neural_log_combining_function("edge_combining_function_2d:sparse")
def edge_combining_function_2d_sparse(a, sp_b):
    """
    Combines the tensors `a` and `sp_b`, where `sp_b` is a sparse tensor.

    :param a: the tensor a
    :type a: tf.Tensor
    :param sp_b: the tensor b
    :type sp_b: tf.SparseTensor
    :return: a combination of the tensors
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
    result = tf.math.divide_no_nan(sum_a, non_zero)
    return tf.reshape(result, [-1, 1])


@neural_log_literal_function("sum")
def summation(a):
    """
    Returns the sum of the values of `a` for each row.

    :param a: the input tensor
    :type a: tf.Tensor
    :return: the result tensor
    :rtype: tf.Tensor
    """
    result = tf.math.reduce_sum(a, axis=1, keepdims=False)
    return tf.reshape(result, [-1, 1])


@neural_log_literal_function("normalize")
def normalize(a):
    """
    Returns the normalized vector `a / sum(a)`.

    :param a: the input vector
    :type a: tf.Tensor
    :return: the normalized vector
    :rtype: tf.Tensor
    """
    return tf.math.divide_no_nan(a, tf.reduce_sum(a))


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


@neural_log_literal_function("pad_d1")
def pad_d1(a, before=0, after=0, value=0):
    """
    Pads the tensor `a` at the first dimension with `before` times `value`
    before the tensor and `after` times `value` after the tensor.

    :param a: the tensor
    :type a: tf.Tensor
    :param before: the length of the padding before the tensor
    :type before: int
    :param after: the length of the padding after the tensor
    :type after: int
    :param value: the constant value
    :type value: int or float
    :return: the padded tensor
    :rtype: tf.Tensor
    """
    padding = tf.constant([[before, after], [0, 0]])
    return tf.pad(a, padding, constant_values=value)


@neural_log_literal_function("pad_d2")
def pad_d2(a, before=0, after=0, value=0):
    """
    Pads the tensor `a` at the second dimension with `before` times `value`
    before the tensor and `after` times `value` after the tensor.

    :param a: the tensor
    :type a: tf.Tensor
    :param before: the length of the padding before the tensor
    :type before: int
    :param after: the length of the padding after the tensor
    :type after: int
    :param value: the constant value
    :type value: int or float
    :return: the padded tensor
    :rtype: tf.Tensor
    """
    padding = tf.constant([[0, 0], [before, after]])
    return tf.pad(a, padding, constant_values=value)


@neural_log_combining_function("embedding_lookup")
def embedding_lookup(ids, params):
    """
    Returns the embeddings lookups.

    The difference of this function to TensorFlow's function is that this
    function expects the ids as the first argument and the parameters as the
    second; while, in TensorFlow, is the other way around.

    :param ids: the ids
    :type ids: tf.Tensor
    :param params: the parameters
    :type params: tf.Tensor
    :return: the lookup
    :rtype: tf.Tensor
    """
    return tf.nn.embedding_lookup(params, ids)


@neural_log_literal_function("partial")
class Partial:
    """
    Defines a literal function that call another function passing the values
    as parameters.
    """

    def __init__(self, function_name, mode="before", **kwargs):
        """
        Creates the Partial function.

        :param function_name: the name of the literal function
        :type function_name: str
        :param mode: the mode of the function. If `before`, the arguments of
        the call function will came before the arguments of the constructor;
        otherwise, it will came after.
        :type mode: str
        :param kwargs: the arguments of the constructor
        :type kwargs: Any
        """
        self._function = self._get_function(function_name)
        self.mode = mode
        self.kwargs = kwargs

    @staticmethod
    def _get_function(function_name):
        if function_name.startswith("tf."):
            names = function_name.split(".")[1:]
            func = tf
            for name in names:
                func = getattr(func, name)
            return func

        return get_literal_function(function_name)

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        if self.mode.lower() == "before":
            new_kwargs = dict(self.kwargs)
            new_kwargs.update(kwargs)
            return self._function(inputs, **new_kwargs)
        else:
            new_kwargs = dict(kwargs)
            new_kwargs.update(self.kwargs)
            return self._function(inputs, **new_kwargs)

    __call__ = call


def build_input_for_terms(terms, cache):
    """
    Creates the input for the edge.

    :param terms: the terms or and edge
    :type terms: collections.Iterable[Term] or Edge
    :param cache: the cache of tensors
    :type cache: Dict[Term, Tensor]
    :return: the inputs for the edge
    :rtype: Tensor or List[Tensor]
    """
    inputs = []
    if isinstance(terms, Edge):
        terms = terms.get_input_terms()
    for term in terms:
        inputs.append(cache[term])

    if len(inputs) == 1:
        return inputs[0]

    return inputs


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

    def is_empty(self):
        """
        Checks if a Layer is empty.

        :return: `True`, if the layer is empty; `False`, otherwise
        :rtype: bool
        """
        return False


class EmptyLayer(NeuralLogLayer):
    """
    Represents an EmptyLayer.
    """

    def __init__(self, name, **kwargs):
        super(EmptyLayer, self).__init__(name, **kwargs)
        self.zero = tf.constant(0.0)

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        return tf.multiply(inputs, self.zero)

    # noinspection PyMissingOrEmptyDocstring
    # def compute_output_shape(self, input_shape):
    #     return tf.TensorShape(input_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(EmptyLayer, self).get_config()

    # noinspection PyMissingOrEmptyDocstring,PyShadowingNames
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def is_empty(self):
        """
        Checks if a Layer is empty.

        The EmptyLayer is always empty.

        :return: `True`, if the layer is empty; `False`, otherwise
        :rtype: bool
        """
        return True

    def __hash__(self):
        return hash("empty")

    def __eq__(self, other):
        return isinstance(other, EmptyLayer)


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
    # def compute_output_shape(self, input_shape):
    #     return tf.TensorShape(input_shape)

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
        if not isinstance(other, FactLayer):
            return False

        return self.kernel is other.kernel and \
               self.fact_combining_function == other.fact_combining_function

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        if inputs is None:
            return None
        kernel = self.get_kernel()
        inputs_shape = inputs.shape
        if len(kernel.shape) > 1:
            # if len(inputs_shape) == 1:
            #     inputs = tf.reshape(inputs, [1, -1])
            if len(inputs_shape) == 2 and \
                    inputs_shape[0] == inputs_shape[1] == 1 and \
                    kernel.shape[0] > 1:
                shape = list(inputs_shape)
                shape[-1] = kernel.shape[0]
                inputs = tf.broadcast_to(inputs, shape)

        return self.fact_combining_function(inputs, kernel)


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
        kernel = tf.linalg.tensor_diag_part(kernel)
        super(DiagonalFactLayer, self).__init__(
            name, kernel, fact_combining_function, **kwargs)


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
        self.kernel_rank = self.kernel.shape.rank
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
        if self.kernel_rank == 1:
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
                 input_constants=None,
                 input_combining_functions=None, output_constant=None,
                 output_extract_function=None, **kwargs):
        """
        Creates a PredicateLayer.

        :param fact_layer: a fact layer
        :type fact_layer: AbstractFactLayer
        :param fact_combining_function: the fact combining function
        :type fact_combining_function: function
        :param input_constants: the input constants
        :type input_constants: tf.Tensor or list[tf.Tensor]
        :param input_combining_functions: the function to combine the fixed
        input with the input of the layer
        :type input_combining_functions: function or list[function]
        :param output_constant: the output constant, if any
        :type output_constant: tf.Tensor
        :param output_extract_function: the function to extract the fact value
        of the fixed output constant
        :type output_extract_function: function
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        self.fact_layer = fact_layer

        self.output_constant = output_constant
        self.output_extract_function = output_extract_function
        self.input_constants = None
        self.input_combining_functions = None
        self.call = self.call_single_input
        self._build_inputs(input_constants, input_combining_functions)

        super(SpecificFactLayer, self).__init__(name, **kwargs)

    def _build_inputs(self, input_constants, input_combining_functions):
        if input_constants is None:
            return
        if isinstance(input_constants, list):
            if len(input_constants) == 0:
                return
            elif len(input_constants) == 1:
                self.input_constants = input_constants[0]
                if isinstance(input_combining_functions, list):
                    self.input_combining_functions = \
                        input_combining_functions[0]
                else:
                    self.input_combining_functions = input_combining_functions
            else:
                self.input_constants = tuple(input_constants)
                self.call = self.call_multiple_inputs
                if isinstance(input_combining_functions, list):
                    self.input_combining_functions = \
                        tuple(input_combining_functions)
                else:
                    self.input_combining_functions = \
                        (self.input_combining_functions,) * len(input_constants)
        else:
            self.input_constants = input_constants
            if isinstance(input_combining_functions, list):
                self.input_combining_functions = input_combining_functions[0]
            else:
                self.input_combining_functions = input_combining_functions

    # noinspection PyMissingOrEmptyDocstring,PyUnusedLocal
    def call_multiple_inputs(self, inputs, **kwargs):
        input_constants = []
        for i in range(len(self.input_constants)):
            input_constants.append(
                self.input_combining_functions[i](
                    self.input_constants[i], inputs[i])
            )
        result = self.fact_layer.call(tuple(input_constants))
        return self._extract_output(result)

    # noinspection PyMissingOrEmptyDocstring,PyUnusedLocal
    def call_single_input(self, inputs, **kwargs):
        if self.input_constants is None:
            input_constant = inputs
        else:
            # noinspection PyCallingNonCallable
            input_constant = self.input_combining_functions(
                self.input_constants, inputs)
        result = self.fact_layer.call(input_constant)
        return self._extract_output(result)

    def _extract_output(self, result):
        if self.output_constant is not None:
            if len(result.shape.as_list()) == 2:
                result = tf.transpose(result)
            result = self.output_extract_function(result, self.output_constant)
            result = tf.reshape(result, [-1, 1])
        return result

    def __hash__(self):
        return hash((self.fact_layer,
                     self.input_combining_functions,
                     self.output_extract_function))

    # noinspection DuplicatedCode
    def __eq__(self, other):
        if not isinstance(other, SpecificFactLayer):
            return False

        if self.fact_layer != other.fact_layer:
            return False

        if self.input_constants is not other.input_constants:
            return False

        if self.output_constant is not other.output_constant:
            return False

        if self.input_combining_functions != other.input_combining_functions:
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
        self._is_empty = self._compute_empty()
        super(LiteralLayer, self).__init__(name, **kwargs)

    def _compute_empty(self):
        for layer in self.input_layers:
            if not layer.is_empty():
                return False
        return True

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        if len(self.input_layers) == 1:
            result = self.input_layers[0](inputs)
            if self.negation_function is not None:
                return self.negation_function(result)
            return result

        result = self.input_layers[0](inputs)
        for input_layer in self.input_layers[1:]:
            if input_layer.is_empty():
                continue
            layer_result = input_layer(inputs)
            result = self.literal_combining_function(result, layer_result)
        if self.negation_function is not None:
            return self.negation_function(result)
        return result

    def is_empty(self):
        """
        Checks if a Layer is empty.

        A literal layer is only empty if all input layers of the literal are
        empty.

        :return: `True`, if the layer is empty; `False`, otherwise
        :rtype: bool
        """
        return self._is_empty

    # noinspection PyMissingOrEmptyDocstring
    # def compute_output_shape(self, input_shape):
    #     return tf.TensorShape(input_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(LiteralLayer, self).get_config()

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __hash__(self):
        if self.negation_function is None and len(self.input_layers) == 1:
            return hash(self.input_layers[0])

        return hash(tuple(self.input_layers) +
                    (self.literal_combining_function, self.negation_function))

    def __eq__(self, other):
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
        :type function: callable
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
    # def compute_output_shape(self, input_shape):
    #     if hasattr(self.function, "compute_output_shape"):
    #         return self.function.compute_output_shape(input_shape)
    #     return tf.TensorShape(input_shape)

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
    # def compute_output_shape(self, input_shape):
    #     return tf.TensorShape(input_shape)

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
        self._is_empty = self._compute_empty()
        super(RuleLayer, self).__init__(name, **kwargs)

    def _compute_empty(self):
        for path in self.paths:
            for layer in path:
                if layer.is_empty():
                    return True
        return False

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

    def is_empty(self):
        """
        Checks if a Layer is empty.

        A RuleLayer is empty if there is, at least, a path with a empty layer
        in it.

        If there is no paths in the rule layer, it is not empty.

        :return: `True`, if the layer is empty; `False`, otherwise
        :rtype: bool
        """
        return self._is_empty

    # noinspection PyMissingOrEmptyDocstring
    # def compute_output_shape(self, input_shape):
    #     return tf.TensorShape(input_shape)

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


class CyclicRuleException(Exception):
    """
    Represents a cyclic rule exception.
    """

    def __init__(self, clause):
        """
        Creates a cyclic rule exception.

        :param clause: the clause that originated the layer
        :type clause: HornClause
        """
        if clause.provenance is None:
            message = "Clause `{}` is cyclic. Clauses must not contain " \
                      "cycles.".format(clause)
        else:
            message = "Clause `{}`, defined in file: `{}` " \
                      "at {} is cyclic. Clauses must not contain " \
                      "cycles.".format(clause, clause.provenance.filename,
                                       clause.provenance.start_line)
        super(CyclicRuleException, self).__init__(message)


def is_any_edge(edge):
    """
    Returns `True` if the edge represents an any literal.

    :param edge: the edge
    :type edge: Edge
    :return: `True`, if the edge represents an any literal; otherwise,
    `False`
    :rtype: bool
    """
    return edge.literal.predicate.name == ANY_PREDICATE_NAME


class GraphRuleLayer(NeuralLogLayer):
    """
    A Layer to represent a logic graph rule.
    """

    def __init__(self, clause, name, rule_graph, literal_layers,
                 grounded_layers,
                 path_combining_function, and_combining_function,
                 neutral_element, **kwargs):
        """
        Creates a GraphRuleLayer.

        :param clause: the horn clause
        :param clause: HornClause
        :param name: the name of the layer
        :type name: str
        :param rule_graph: the rule graph
        :type rule_graph: RuleGraph
        :param literal_layers: the literal layers
        :type literal_layers: Dict[Edge, LiteralLayer]
        :param grounded_layers: the grounded literal layers
        :type grounded_layers: List[LiteralLayer]
        :param path_combining_function: the path combining function for the
        destination
        :type and_combining_function: function
        :param and_combining_function: the path combining function for the
        intermediary paths
        :type path_combining_function: function
        :param neutral_element: the neural element to be passed to the
        grounded layer
        :type neutral_element: tf.Tensor
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        self.clause = clause
        self.rule_graph = data_structures.NoDependency(rule_graph)
        self.destinations = rule_graph.destinations
        self.literal_layers = dict()
        for key, value in literal_layers.items():
            self.literal_layers[str(key)] = value
        self.grounded_layers = grounded_layers
        self.path_combining_function = path_combining_function
        self.and_combining_function = and_combining_function
        self.neutral_element = neutral_element
        self._is_empty = self._compute_empty()
        self.cache = dict()
        super(GraphRuleLayer, self).__init__(name, **kwargs)

    def _compute_empty(self):
        for layer in self.literal_layers.values():
            if layer.is_empty():
                return True
        return False

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        self.cache = dict()
        results = []
        for destination in self.destinations:
            results.append(self._compute_term(inputs, destination))
        path_result = reduce(self.path_combining_function, results)
        if path_result is None:
            path_result = self.neutral_element

        for grounded_layer in self.grounded_layers:
            grounded_result = grounded_layer(self.neutral_element)
            path_result = self.path_combining_function(path_result,
                                                       grounded_result)
        return path_result

    # TODO: refactor this method
    # TODO: compute de returning paths
    def _compute_term(self, inputs, term):
        terms_to_compute = deque([term])  # type: deque[Term]
        while len(terms_to_compute) > 0:
            size = len(terms_to_compute)
            for i in range(size):
                current = terms_to_compute.pop()

                # The term is cached
                tensor = self.cache.get(current, None)
                if tensor is not None:
                    continue
                not_any_tensor = True
                # noinspection PyUnresolvedReferences
                if current in self.rule_graph.sources:
                    # The term is an input term
                    # noinspection PyUnresolvedReferences
                    if len(self.rule_graph.sources) == 1:
                        tensor = inputs
                    else:
                        # noinspection PyUnresolvedReferences
                        tensor = inputs[self.rule_graph.sources.index(current)]
                else:
                    # The term has to be compute
                    if self._append_terms_to_compute(current, terms_to_compute):
                        continue
                    tensors = []
                    # noinspection PyUnresolvedReferences
                    edges = self.rule_graph.input_edges_by_nodes.get(
                        current, None)
                    if edges is None:
                        tensors.append(self.neutral_element)
                    else:
                        for edge in edges:
                            if is_any_edge(edge):
                                if current == term:
                                    tensor = self._compute_any_predicate(
                                        current, edge)
                                    tensors.append(tensor)
                                    not_any_tensor = False
                                else:
                                    tensors.append(self.neutral_element)
                            else:
                                literal_layer = self.literal_layers[str(edge)]
                                edge_inputs = build_input_for_terms(
                                    edge, self.cache)
                                temp_tensor = literal_layer(edge_inputs)
                                tensors.append(temp_tensor)

                    # Combining the different paths
                    tensor = self._combine_paths(current, term, tensors)

                # Compute loops
                if not_any_tensor:
                    tensor = self.compute_loop_for_term(current, tensor)

                # Add tensor to cache
                self.cache[current] = tensor

        return self.cache[term]

    def _combine_paths(self, current, destination, tensors):
        """
        Combines the tensors of the path

        :param current: the current term
        :type current: Term
        :param destination: the destination term
        :type destination: Term
        :param tensors: the list of tensors to combine
        :type tensors: list[tf.Tensor]
        :return: the combined tensor
        :rtype: tf.Tensor
        """
        if current == destination:
            combining_function = self.path_combining_function
        else:
            combining_function = self.and_combining_function
        return reduce(combining_function, tensors)

    def _compute_any_predicate(self, current, edge):
        """
        Computes the any predicate.

        :param current: the current term
        :type current: Term
        :param edge: the current edge
        :type edge: Edge
        :return: the result of the predicate
        :rtype: tf.Tensor
        """
        temp_tensors = []
        for d_term in edge.get_input_terms():
            temp_tensor = self._get_any_layer(
                d_term, current)(self.cache[d_term])
            temp_tensors.append(temp_tensor)
        dest_tensor = tf.reshape(self.neutral_element, [1, -1])
        dest_tensor = self.compute_loop_for_term(current, dest_tensor)
        temp_tensors.append(dest_tensor)
        return reduce(tf.math.multiply, temp_tensors)

    def _append_terms_to_compute(self, current, terms_to_compute):
        """
        Appends the terms to compute.

        :param current: the current term
        :type current: Term
        :param terms_to_compute: the previous terms to compute
        :type terms_to_compute: deque[Term]
        :return: `True`, if it appended any term; otherwise, `False`.
        :rtype: bool
        """
        has_term_to_compute = False
        # noinspection PyUnresolvedReferences
        for edge in self.rule_graph.input_edges_by_nodes.get(current, []):
            for input_term in edge.get_input_terms():
                if input_term not in self.cache:
                    if not has_term_to_compute:
                        if current in terms_to_compute:
                            raise CyclicRuleException(self.clause)
                        terms_to_compute.append(current)
                        has_term_to_compute = True
                    if input_term not in terms_to_compute:
                        terms_to_compute.append(input_term)
        return has_term_to_compute

    def compute_loop_for_term(self, term, inputs):
        """
        Computes the loops for the term.

        :param term: the term
        :type term: `Term`
        :param inputs: the inputs of the term
        :type inputs: tf.Tensor
        :return: the tensor result after compute the loops
        :rtype: tf.Tensor
        """
        tensor = inputs
        # noinspection PyUnresolvedReferences
        for loop in self.rule_graph.loops_by_nodes.get(term, []):
            tensor = self.literal_layers[str(loop)](tensor)
        return tensor

    def _get_any_layer(self, in_term, out_term):
        """
        Gets the any literal layer for the terms.

        :param in_term: the input term
        :type in_term: Term
        :param out_term: the output term
        :type out_term: Term
        :return: the any literal layer
        :rtype: LiteralLayer
        """
        any_literal = Literal(Atom(ANY_PREDICATE_NAME, in_term, out_term))
        edge = Edge(any_literal, (0,), 1)
        return self.literal_layers[str(edge)]

    def is_empty(self):
        """
        Checks if a Layer is empty.

        A RuleLayer is empty if there is, at least, a path with a empty layer
        in it.

        If there is no paths in the rule layer, it is not empty.

        :return: `True`, if the layer is empty; `False`, otherwise
        :rtype: bool
        """
        return self._is_empty

    # noinspection PyMissingOrEmptyDocstring
    # def compute_output_shape(self, input_shape):
    #     return tf.TensorShape(input_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(GraphRuleLayer, self).get_config()

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DiagonalRuleLayer(NeuralLogLayer):
    """
    Class to extract the value of rules with same variables in the head.
    """

    def __init__(self, rule_layer, combining_function, **kwargs):
        """
        Creates a DiagonalRuleLayer.

        :param name: the name of the layer
        :type name: str
        :param rule_layer: the rule layer
        :type rule_layer: RuleLayer
        :param combining_function: the combining function
        :type combining_function: function
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        name = rule_layer.name + "_diagonal"
        self.rule_layer = rule_layer
        self.combining_function = combining_function
        super(DiagonalRuleLayer, self).__init__(name, **kwargs)

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        result = self.rule_layer(inputs)
        return self.combining_function(inputs, result)

    # noinspection PyMissingOrEmptyDocstring
    # def compute_output_shape(self, input_shape):
    #     return self.rule_layer.compute_output_shape(input_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(DiagonalRuleLayer, self).get_config()

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __hash__(self):
        return hash(("_diagonal", self.rule_layer))

    # noinspection DuplicatedCode
    def __eq__(self, other):
        if not isinstance(other, DiagonalRuleLayer):
            return False

        if self.rule_layer != other.rule_layer:
            return False

        if self.combining_function is not other.combining_function:
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
    # def compute_output_shape(self, input_shape):
    #     output_shape = list(input_shape)
    #     output_shape[-1] = 1
    #     return tuple(output_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(ExtractUnaryLiteralLayer, self).get_config()

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __hash__(self):
        return hash(("_extract_unary", self.literal_layer))

    # noinspection DuplicatedCode
    def __eq__(self, other):
        if not isinstance(other, ExtractUnaryLiteralLayer):
            return False

        if self.literal_layer != other.literal_layer:
            return False

        if self.input_combining_function is not other.input_combining_function:
            return False

        return True
