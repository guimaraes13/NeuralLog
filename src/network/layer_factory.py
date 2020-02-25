"""
A factory of layers.
"""

import logging
import re
from enum import Enum
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix

from src.knowledge.program import NeuralLogProgram
from src.language.language import Predicate, Atom, Variable, \
    get_variable_atom, get_renamed_atom
from src.network.network_functions import get_initializer, \
    get_combining_function, SPARSE_FUNCTION_SUFFIX, FactLayer, \
    AttributeFactLayer, SpecificFactLayer, DiagonalFactLayer, \
    InvertedFactLayer, InvertedSpecificFactLayer

logger = logging.getLogger()

SPACE_REGEX = re.compile(r"\s")
ALLOW_REGEX = re.compile(r'[^a-zA-Z0-9\-_]')
FIRST_CHARACTER = re.compile(r"[A-Za-z0-9.]")


def decorate_factory_function():
    """
    Decorates the function to handle the atom.

    :return: a function to registry the function
    :rtype: function
    """
    commands = {}

    def decorator(function_key):
        """
        Returns a function to register the command with the tensor function key.

        :param function_key: the tensor function key
        :type function_key: TensorFunctionKey
        :return: a function to register the command
        :rtype: function
        """

        def registry(func):
            """
            Registries the function as a command to handle the builtin
            predicate.

            :param func: the function to be registered.
            :type func: function
            :return: the registered function
            :rtype: function
            """
            commands[function_key] = func
            return func

        return registry

    decorator.functions = commands
    return decorator


def get_initial_value_by_name(initializer, shape):
    """
    Gets the initializer by name or config.

    :param initializer: the initializer name or config
    :type initializer: str, dict
    :param shape: the shape of the initial value
    :type shape: int or collections.Iterable[int]
    :return: the initial value
    :rtype: function
    """
    initializer = get_initializer(initializer)
    initial_value = lambda: initializer(shape)
    return initial_value


class NotGroundAtomException(Exception):
    """
    Represents an atom malformed exception.
    """

    def __init__(self, atom) -> None:
        """
        Creates a not ground atom exception.

        :param atom: the not ground atom
        :type atom: Atom
        """
        super().__init__("Atom {} is not ground.".format(atom))


class FactoryTermType(Enum):
    """
    Defines the types of a term.
    """
    CONSTANT = 0
    ITERABLE_CONSTANT = 1
    VARIABLE = 2
    NUMBER = 3


class TensorFunctionKey:
    """
    Defines the key of the function to build the tensor.
    """

    def __init__(self, arity, true_arity, trainable, *args):
        self.arity = arity
        self.true_arity = true_arity
        self.trainable = trainable
        if len(args) == 0:
            self.terms = tuple()
        else:
            self.terms = tuple(args)

    def key(self):
        """
        Specifies the keys to be used in the equals and hash functions.
        :return: the keys
        :rtype: Any
        """
        return self.arity, self.true_arity, self.trainable, self.terms

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.key() == other.key()
        return False


def get_standardised_name(string):
    """
    Gets a standardised name, from the string, that can be used as a variable
    name in TensorFlow. The standardised name containing only the values that
    matches: `[A-Za-z0-9.][A-Za-z0-9_.\\-/]*`

    :param string: the name
    :type string: str
    :return: The standardised name.
    :rtype: str
    """
    standard = SPACE_REGEX.sub("_", string)
    standard = ALLOW_REGEX.sub("-", standard)
    # if FIRST_CHARACTER.search(standard[0]) is not None:
    #     return "." + standard

    return standard


class LayerFactory:
    """
    A factory of Keras Layers.
    """

    SPARSE_THRESHOLD = 0.3

    variable_cache: Dict[Atom, tf.Tensor] = dict()
    "Caches the variable tensors of the atoms"

    _tensor_by_name: Dict[str or tuple, tf.Tensor] = dict()
    "The tensors by name"

    _matrix_representation: Dict[Predicate, Any] = dict()
    "Caches the matrices representations."

    _vector_representation: Dict[Atom, Any] = dict()
    "Caches the vector representations of binary predicates with constant."

    _diagonal_matrix_representation: Dict[Predicate, csr_matrix] = dict()
    "Caches the diagonal matrices representations."

    tensor_function = decorate_factory_function()

    program: NeuralLogProgram

    def __init__(self, program, layer_name_format="fact_layer_{}", train=True):
        """
        Creates a TensorFactory.

        :param layer_name_format: the format of the layer name
        :type layer_name_format: str
        :param train: if `False`, all the literals will be considered as not
        trainable/learnable, this is useful to build neural networks for
        inference only. In this way, the unknown facts will be treated as
        zeros, instead of being randomly initialized
        :type train: bool
        :param program: the NeuralLog program
        :type program: NeuralLogProgram
        """
        self.program = program
        # noinspection PyUnresolvedReferences
        self.function = self.tensor_function.functions
        self.train = train
        # Layer Properties
        self.layer_name_format = layer_name_format

    def build_atom(self, atom):
        """
        Builds the matrix representation for the atom.

        :param atom: the atom
        :type atom: Atom
        :raise UnsupportedMatrixRepresentation in the case the predicate is
        not convertible to matrix form
        :return: the matrix representation of the data for the given predicate
        :rtype: AbstractFactLayer
        """
        key = self.get_atom_key(atom)
        return self.function[key](self, atom)

    def get_atom_key(self, atom):
        """
        Gets the key of the atom, used to get the correct function to handle it.

        :param atom: the atom
        :type atom: Atom
        :return: the key of the atom
        :rtype: TensorFunctionKey
        """
        term_types = []
        types = self.program.predicates[atom.predicate]
        for i in range(atom.arity()):
            if types[i].number:
                term_types.append(FactoryTermType.NUMBER)
            elif atom.terms[i].is_constant():
                # if atom.terms[i] in self.program.iterable_constants.inverse:
                if self.program.is_iterable_constant(atom, i):
                    term_types.append(FactoryTermType.ITERABLE_CONSTANT)
                else:
                    term_types.append(FactoryTermType.CONSTANT)
            else:
                term_types.append(FactoryTermType.VARIABLE)
        trainable = self.train and \
                    atom.predicate in self.program.trainable_predicates
        key = TensorFunctionKey(atom.arity(),
                                self.program.get_true_arity(atom.predicate),
                                trainable, *term_types)
        return key

    def get_matrix_representation(self, predicate):
        """
        Gets the matrix representation for the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :raise UnsupportedMatrixRepresentation in the case the predicate is
        not convertible to matrix form
        :return: the matrix representation of the data for the given predicate
        :rtype: csr_matrix or np.matrix or (csr_matrix, csr_matrix) or float
        """
        return self._matrix_representation.setdefault(
            predicate, self.program.get_matrix_representation(predicate))

    def get_vector_representation_with_constant(self, atom):
        """
        Gets the vector representation for the binary predicate with constant.

        :param atom: the atom
        :type atom: Atom
        :raise UnsupportedMatrixRepresentation in the case the predicate is
        not convertible to matrix form
        :return: the vector representation of the data for the given atom
        :rtype: csr_matrix
        """
        return self._vector_representation.setdefault(
            atom, self.program.get_vector_representation_with_constant(atom))

    def get_diagonal_matrix_representation(self, predicate):
        """
        Gets the diagonal matrix representation for a binary predicate.

        :param predicate: the binary predicate
        :type predicate: Predicate
        :return: the matrix representation of the data
        :rtype: csr_matrix
        """
        return self._diagonal_matrix_representation.setdefault(
            predicate,
            self.program.get_diagonal_matrix_representation(predicate))

    def get_allow_sparse(self, predicate=None):
        """
        Gets the allow sparse option for the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the allow sparse option
        :rtype: bool
        """
        return self.program.get_parameter_value("allow_sparse", predicate)

    def get_weighted_attribute_combining_function(self, predicate=None):
        """
        Gets the attribute combining function. This is the function to
        combine the weights and values of the attribute facts.

        The default function is the `tf.math.multiply`.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the combining function
        :rtype: function
        """
        combining_function = self.program.get_parameter_value(
            "weighted_attribute_combining_function", predicate)
        return get_combining_function(combining_function)

    def get_attributes_combining_function(self, predicate=None):
        """
        Gets the attribute combining function. This is the function to
        combine the numeric terms of a fact.

        The default function is the `tf.math.multiply`.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the combining function
        :rtype: function
        """
        combining_function = self.program.get_parameter_value(
            "attributes_combine_function", predicate)
        return get_combining_function(combining_function)

    def get_and_combining_function(self, predicate=None):
        """
        Gets the AND combining function. This is the function to combine
        different vector and get an `AND` behaviour between them.

        The default is to multiply all the paths, element-wise, by applying the
        `tf.math.multiply` function.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the combining function
        :rtype: function
        """
        combining_function = self.program.get_parameter_value(
            "and_combining_function", predicate)
        return get_combining_function(combining_function)

    def get_edge_combining_function(self, predicate=None):
        """
        Gets the edge combining function. This is the function to extract the
        value of the fact based on the input.

        The default is the element-wise multiplication implemented by the
        `tf.math.multiply` function.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the combining function
        :rtype: function
        """
        combining_function = self.program.get_parameter_value(
            "edge_combining_function", predicate)
        return get_combining_function(combining_function)

    def get_edge_combining_function_2d(self, predicate=None, sparse=False):
        """
        Gets the edge combining function 2d. This is the function to extract the
        value of the fact based on the input, for 2d facts.

        The default is the dot multiplication implemented by the `tf.matmul`
        function.

        :param predicate: the predicate
        :type predicate: Predicate
        :param sparse: if the kernel tensor is sparse
        :param sparse: bool
        :return: the combining function
        :rtype: function
        """
        function_name = "edge_combining_function_2d"
        if sparse:
            function_name += SPARSE_FUNCTION_SUFFIX
        combining_function = self.program.get_parameter_value(function_name,
                                                              predicate)
        return get_combining_function(combining_function)

    def get_attribute_edge_combining_function(self, predicate=None):
        """
        Gets the edge combining function for attribute predicates.
        This is the function to extract the value of the fact based on the
        input, for attribute facts.

        The default is the dot multiplication implemented by the `tf.matmul`
        function.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the combining function
        :rtype: function
        """
        function_name = "attribute_edge_combining_function"
        combining_function = self.program.get_parameter_value(function_name,
                                                              predicate)
        return get_combining_function(combining_function)

    def get_invert_fact_function(self, predicate, sparse=False):
        """
        Gets the fact inversion function. This is the function to extract
        the inverse of the facts.

        The default is the transpose function implemented by `tf.transpose`.

        :param predicate: the predicate
        :type predicate: Predicate
        :param sparse: if the kernel tensor is sparse
        :param sparse: bool
        :return: the combining function
        :rtype: function
        """
        function_name = "invert_fact_function"
        if sparse:
            function_name += SPARSE_FUNCTION_SUFFIX
        combining_function = self.program.get_parameter_value(
            function_name, predicate)
        return get_combining_function(combining_function)

    def get_output_extract_function(self, predicate):
        """
        Gets the output extract function. This is the function to extract the
        value of an atom with a constant at the last term position.

        The default function is the `tf.nn.embedding_lookup`.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the combining function
        :rtype: function
        """
        combining_function = self.program.get_parameter_value(
            "output_extract_function", predicate)
        return get_combining_function(combining_function)

    # noinspection PyTypeChecker
    def _get_variable(self, name, value, shape, dtype=tf.float32):
        tensor = self._tensor_by_name.get(name, None)
        if tensor is None:
            tensor = tf.Variable(initial_value=value, dtype=dtype,
                                 shape=shape, name=get_standardised_name(name),
                                 trainable=True)
            self._tensor_by_name[name] = tensor
        return tensor

    def _matrix_to_variable(self, atom, initial_value=None, shape=None,
                            name_format=None):
        """
        Returns a variable representation of the atom.

        :param atom: the atom
        :type atom: Atom
        :param initial_value: the initial value
        :type initial_value: np.array or csr_matrix or float or function
        :param shape: the shape of the variable
        :type shape: Any
        :return: the tensor representation of the atom
        :rtype: tf.Tensor or tf.SparseTensor
        """
        renamed_atom = get_renamed_atom(atom)
        tensor = self.variable_cache.get(renamed_atom, None)
        if tensor is None:
            if initial_value is None:
                initial_value = self.get_matrix_representation(atom.predicate)
            if isinstance(initial_value, csr_matrix):
                initial_value = initial_value.todense()
            if isinstance(initial_value, np.ndarray):
                initial_value = np.squeeze(np.array(initial_value))
            if shape is None:
                if hasattr(initial_value, 'shape'):
                    shape = list(initial_value.shape)
                else:
                    shape = []

            if name_format is None:
                name = renamed_atom.__str__()
            else:
                name = name_format.format(renamed_atom.__str__())

            tensor = self._get_variable(name, initial_value, shape)
            # noinspection PyTypeChecker
            self.variable_cache[renamed_atom] = tensor

        return tensor

    def _matrix_to_variable_with_value(self, atom, initial_value, shape=None,
                                       constant=None):
        """
        Returns a variable representation of the atom, initializing the
        unknown values with initial value.

        :param atom: the atom
        :type atom: Atom
        :param initial_value: the initial value
        :type initial_value: np.array or csr_matrix or float or function
        :param shape: the shape of the variable
        :type shape: Any
        :param constant: the constant value of the atom. If not `None`,
        the mask will be zero for values different from constant
        :type constant: float or int
        :return: the tensor representation of the atom
        :rtype: tf.Tensor or tf.SparseTensor
        """
        renamed_atom = get_renamed_atom(atom)
        tensor = self.variable_cache.get(renamed_atom, None)
        if tensor is None:
            if shape is None:
                if hasattr(initial_value, 'shape'):
                    shape = list(initial_value.shape)
                else:
                    shape = []
            if self.program.get_true_arity(atom.predicate) == 2 and \
                    (atom.terms[0].is_constant() or
                     atom.terms[1].is_constant()):
                mask = self.program.get_vector_representation_with_constant(
                    atom, mask=True)
                weights = self.get_vector_representation_with_constant(atom)
            else:
                mask = self.program.get_matrix_representation(
                    atom.predicate, mask=True)
                weights = self.get_matrix_representation(atom.predicate)
            if isinstance(mask, csr_matrix):
                mask = mask.todense()
            mask = 1.0 - mask
            if isinstance(weights, tuple):
                values = weights[1]
                weights = weights[0].todense()
                if constant is not None:
                    for i in range(len(mask)):
                        if not np.isclose(values[i, 0], constant):
                            mask[i, 0] = 1.0
                            weights[i, 0] = 0.0
            else:
                weights = weights.todense()

            # noinspection PyTypeChecker
            tensor = self._get_variable(
                "init-{}".format(renamed_atom.__str__()), initial_value, shape)
            mask = self._build_constant(
                mask, shape, "mask-{}".format(renamed_atom.__str__()))
            tensor = tf.multiply(tensor, mask)
            w = self._get_variable("weights-{}".format(renamed_atom.__str__()),
                                   weights, weights.shape)
            if tuple(shape) != weights.shape:
                w = tf.reshape(w, shape)
            tensor = tf.add(tensor, w)
            tensor = self._matrix_to_variable(renamed_atom, tensor, shape)
            # noinspection PyTypeChecker
            self.variable_cache[renamed_atom] = tensor

        return tensor

    def _matrix_to_constant(self, atom, value, shape=None, name_format=None):
        """
        Returns a constant representation of the atom.

        :param atom: the atom
        :type atom: Atom
        :param value: the value of the constant
        :type value: np.array or csr_matrix or float
        :param shape: the shape of the variable
        :type shape: Any
        :param name_format: the name format of the tensor
        :type name_format: str
        :return: the tensor representation of the atom
        :rtype: tf.Tensor or tf.SparseTensor
        """
        renamed_atom = get_renamed_atom(atom)
        if name_format is None:
            name = renamed_atom.__str__()
        else:
            name = name_format.format(renamed_atom.__str__())
        tensor = self._tensor_by_name.get(name, None)
        if tensor is None:
            if shape is None:
                if hasattr(value, 'shape'):
                    shape = list(value.shape)
                else:
                    shape = []
            if name_format is None:
                name = renamed_atom.__str__()
            else:
                name = name_format.format(renamed_atom.__str__())
            allow_sparse = self.get_allow_sparse(renamed_atom.predicate)
            tensor = self._build_constant(value, shape, name,
                                          allow_sparse=allow_sparse)
            # noinspection PyTypeChecker
            self._tensor_by_name[name] = tensor

        return tensor

    def _build_constant(self, value, shape, name, allow_sparse=True):
        """
        Builds the constant for the atom.

        :param value: the value of the constant
        :type value: np.array or csr_matrix or float
        :param shape: the shape of the variable
        :type shape: Any
        :param name: the name of the tensor
        :type name: str
        :return: the tensor representation of the atom
        :rtype: tf.Tensor or tf.SparseTensor
        """
        tensor = self._tensor_by_name.get(name, None)
        if tensor is None:
            if isinstance(value, csr_matrix):
                sparsity = len(value.data) / np.prod(value.shape,
                                                     dtype=np.float32)
                if allow_sparse and sparsity < self.SPARSE_THRESHOLD:
                    data = value.data
                    if len(data) == 0:
                        data = [0.0]
                        rows, columns = [0], [0]
                    else:
                        rows, columns = value.nonzero()
                    if len(shape) == 1:
                        tensor = tf.SparseTensor(
                            indices=list(map(lambda x: list(x), zip(rows))),
                            values=data, dense_shape=shape)
                    else:
                        tensor = tf.SparseTensor(
                            indices=list(map(lambda x: list(x),
                                             zip(rows, columns))),
                            values=data, dense_shape=shape)
                    tensor = tf.sparse.reorder(tensor)
                else:
                    tensor = tf.constant(value=value.todense(),
                                         dtype=tf.float32, shape=shape,
                                         name=get_standardised_name(name))
            else:
                tensor = tf.constant(value=value,
                                     dtype=tf.float32, shape=shape,
                                     name=get_standardised_name(name))
            self._tensor_by_name[name] = tensor

        return tensor

    def _get_initial_value_for_atom(self, atom):
        """
        Gets the initial value for the atom, the atom must be grounded.

        :param atom: the atom, must be grounded
        :type atom: Atom
        :raise NotGroundAtomException if the atom is not ground
        :return: the initial value of the atom
        :rtype: function or float
        """
        if not atom.is_grounded():
            raise NotGroundAtomException(atom)

        value = self.program.facts_by_predicate.get(
            atom.predicate, dict()).get(atom.simple_key(), None)
        if value is None:
            initializer_name = self._get_initializer_name(atom.predicate)
            value = get_initial_value_by_name(initializer_name, [])
            logger.debug("Creating atom %s with initializer %s",
                         atom, initializer_name)
        else:
            value = value.weight

        return value

    def _get_initializer_name(self, predicate):
        """
        Gets the name of the initializer for the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the name of the initializer
        :rtype: str
        """
        return self.program.get_parameter_value("initial_value", predicate)

    def get_constant_lookup(self, atom, term_index):
        """
        Builds the constant lookup for the term in the atom.

        :param atom: the atom
        :type atom: Atom
        :param term_index: the index of the term
        :type term_index: int
        :return: the constant lookup tensor
        :rtype: tf.Tensor
        """
        term = atom.terms[term_index]
        name = get_standardised_name(term.value)
        key = atom.predicate, term_index, name
        tensor = self._tensor_by_name.get(key, None)
        if tensor is None:
            constant = self.program.get_index_of_constant(
                atom.predicate, term_index, term)
            tensor = tf.constant(constant, name=name)
            self._tensor_by_name[key] = tensor
        return tensor

    def get_one_hot_tensor(self, atom, term_index):
        """
        Gets an one-hot row tensor for the iterable constant in the atom.

        :param atom: the atom
        :type atom: Atom
        :param term_index: the index of the term
        :type term_index: int
        :return: the one-hot row tensor
        :rtype: tf.Tensor
        """
        term = atom.terms[term_index]
        return tf.one_hot(
            [self.program.get_index_of_constant(
                atom.predicate, term_index, term)],
            depth=self.program.get_constant_size(atom.predicate, term_index),
            dtype=tf.float32,
            name=get_standardised_name(term.value))

    def _get_layer_name(self, atom):
        return get_standardised_name(self.layer_name_format.format(
            atom.__str__()))

    def _build_simple_fact_layer(self, atom, kernel):
        """
        Builds a simple fact layer from the atom and the kernel.

        :param atom: the atom
        :type atom: Atom
        :param kernel: the kernel
        :type kernel: tf.Tensor
        :return: the fact layer
        :rtype: FactLayer
        """
        name = self._get_layer_name(atom)
        sparse = isinstance(kernel, tf.SparseTensor)
        if kernel.shape.rank == 2:
            combining_func = self.get_edge_combining_function_2d(
                atom.predicate, sparse)
        else:
            if sparse:
                kernel = tf.sparse.to_dense(kernel)
            combining_func = self.get_edge_combining_function(atom.predicate)
        return FactLayer(name, kernel, combining_func)

    # noinspection PyMissingOrEmptyDocstring
    def _get_not_trainable_constants(self, atom):
        fact = self.program.facts_by_predicate.get(
            atom.predicate, dict()).get(atom.simple_key(), None)
        if fact is None:
            weight = 0.0
            if atom.provenance is not None:
                logger.warning(
                    "Warning: there is no fact matching the atom "
                    "%s at line %d:%d, weight replaced by %d.",
                    atom, atom.provenance.start_line,
                    atom.provenance.start_column, weight)
            else:
                logger.warning(
                    "Warning: there is no fact matching the atom "
                    "%s, weight replaced by %d.", atom, weight)
        else:
            weight = fact.weight
        kernel = self._matrix_to_constant(atom, weight)
        return self._build_simple_fact_layer(atom, kernel)

    def _get_arity_x_0_attribute(self, atom, trainable=False):
        weight, value = self.get_matrix_representation(atom.predicate)
        for i in range(atom.arity()):
            if atom.terms[i].is_constant():
                if value[i] != atom.terms[i].value:
                    if trainable:
                        initializer_name = self._get_initializer_name(
                            atom.predicate)
                        weight = get_initial_value_by_name(initializer_name,
                                                           [])
                    else:
                        weight = 0.0
        if atom.arity() == 2 and atom.terms[0] == atom.terms[1]:
            if value[0] != value[-1]:
                weight = 0.0
        if trainable:
            w_tensor = self._matrix_to_variable(atom, weight, shape=[])
        else:
            w_tensor = self._matrix_to_constant(atom, weight, shape=[])
        v_tensor = self._matrix_to_constant(atom, value[0], shape=[],
                                            name_format="v-{}-0")
        for i in range(1, atom.arity()):
            v_tensor_i = self._matrix_to_constant(
                atom, value[i], shape=[],
                name_format="v-{}-{}".format("{}", i))
            v_tensor = self.get_attributes_combining_function(
                atom.predicate)(v_tensor, v_tensor_i)

        name = self._get_layer_name(atom)
        fact_combining_func = \
            self.get_edge_combining_function(atom.predicate)
        weight_combination_func = \
            self.get_weighted_attribute_combining_function(atom.predicate)
        return AttributeFactLayer(name, w_tensor, v_tensor,
                                  fact_combining_func, weight_combination_func)

    def _get_arity_2_1_constant_attribute(self, atom, attribute_index,
                                          trainable=False):
        atom_value = self.program.facts_by_predicate.get(
            atom.predicate, dict()).get(atom.simple_key(), None)
        if atom_value is None:
            if trainable:
                initializer_name = self._get_initializer_name(
                    atom.predicate)
                weight = get_initial_value_by_name(initializer_name, [])
            else:
                weight = 0.0
                logger.warning(
                    "Warning: there is no fact matching the atom "
                    "%s, weight replaced by %d.", atom, weight)
            value = atom.terms[attribute_index].value
        elif not atom.terms[attribute_index].is_constant() or \
                atom.terms[attribute_index] == \
                atom_value.terms[attribute_index]:
            weight = atom_value.weight
            value = atom_value.terms[attribute_index].value
        else:
            if trainable:
                initializer_name = self._get_initializer_name(
                    atom.predicate)
                weight = get_initial_value_by_name(initializer_name, [])
            else:
                weight = 0.0
            value = atom.terms[attribute_index].value

        if trainable:
            w_tensor = self._matrix_to_variable(
                atom, weight, shape=[], name_format="w-{}")
        else:
            w_tensor = self._matrix_to_constant(atom, weight, shape=[],
                                                name_format="w-{}")
        v_tensor = self._matrix_to_constant(atom, value, shape=[],
                                            name_format="v-{}")
        name = self._get_layer_name(atom)
        fact_combining_func = \
            self.get_edge_combining_function(atom.predicate)
        weight_combination_func = \
            self.get_weighted_attribute_combining_function(atom.predicate)
        return AttributeFactLayer(name, w_tensor, v_tensor,
                                  fact_combining_func, weight_combination_func)

    # noinspection PyMissingOrEmptyDocstring
    def _get_weights_and_values_for_attribute_predicate(
            self, atom, trainable=False, constant_value=None):
        weight, value = self.get_matrix_representation(atom.predicate)
        term_index = \
            1 if self.program.predicates[atom.predicate][0].number else 0
        shape = [self.program.get_constant_size(atom.predicate, term_index)]
        if atom.arity() == 2 and atom.terms[0] == atom.terms[1]:
            # This case, we have an attribute fact with the same variable in
            # both position, since one of the variables must be an entity and
            # the other must be a numeric value, and entities and numeric
            # values are disjoint sets, the variables cannot happen to be
            # equal. Thus, it returns zero weight values for every entry.
            if atom.provenance is None:
                logger.warning(
                    "Warning: attribute predicate with same variable in both "
                    "positions. Since the set of entities and the set of "
                    "numeric values are disjoint, it will return zero weight "
                    "for any entry for atom: %s.", atom)
            else:
                logger.warning(
                    "Warning: attribute predicate with same variable in both "
                    "positions. Since the set of entities and the set of "
                    "numeric values are disjoint, it will return zero weight "
                    "for any entry for atom: %s at %d:%d.",
                    atom, atom.provenance.start_line,
                    atom.provenance.start_column)
            weight = csr_matrix((shape[0], 1), dtype=np.float32)
            w_tensor = self._matrix_to_constant(atom, weight.todense(),
                                                shape=shape,
                                                name_format="w:{}")
        elif trainable:
            initializer_name = self._get_initializer_name(atom.predicate)
            initial_value = get_initial_value_by_name(initializer_name, shape)
            w_tensor = self._matrix_to_variable_with_value(
                atom, initial_value, shape=shape, constant=constant_value)
            if constant_value is not None:
                value = np.ones(shape, np.float32) * constant_value
        elif constant_value is not None:
            weight = weight.copy()
            value = value.copy()
            for i in range(len(weight.data)):
                if not np.isclose(value.data[i], constant_value):
                    weight.data[i] = 0.0
                    value.data[i] = constant_value
            w_tensor = self._matrix_to_constant(
                atom, weight.todense(), shape=shape, name_format="w-{}")
        else:
            w_tensor = self._matrix_to_constant(
                atom, weight.todense(), shape=shape, name_format="w-{}")

        if isinstance(value, csr_matrix):
            value = value.todense()

        v_tensor = self._matrix_to_constant(atom, value, shape=shape,
                                            name_format="v:{}")
        return w_tensor, v_tensor

    def _get_arity_2_1_iterable_constant_number(self, atom, constant_index):
        name = self._get_layer_name(atom)
        predicate = atom.predicate
        output_extract_func = self.get_output_extract_function(predicate)
        if constant_index == 0:
            variable_atom = Atom(atom.predicate, "X0", atom.terms[1])
            output_index = 0
        else:
            variable_atom = Atom(atom.predicate, atom.terms[0], "X0")
            output_index = 1
        output_constant = self.get_constant_lookup(atom, output_index)
        fact_layer = self.build_atom(variable_atom)
        return SpecificFactLayer(
            name, fact_layer,
            output_constant=output_constant,
            output_extract_function=output_extract_func
        )

    def _get_arity_2_1_variable_number(self, atom, attribute_index,
                                       trainable=False):
        constant_value = None
        if atom.terms[attribute_index].is_constant():
            constant_value = atom.terms[attribute_index].value
        w_tensor, v_tensor = \
            self._get_weights_and_values_for_attribute_predicate(
                atom, trainable, constant_value=constant_value)
        name = self._get_layer_name(atom)
        fact_combining_func = \
            self.get_edge_combining_function(atom.predicate)
        weight_combination_func = \
            self.get_weighted_attribute_combining_function(atom.predicate)
        return AttributeFactLayer(name, w_tensor, v_tensor,
                                  fact_combining_func, weight_combination_func)

    def _get_arity_2_2_trainable_variable_and_constant(self, atom):
        initializer_name = self._get_initializer_name(atom.predicate)
        if atom.terms[0].is_constant():
            shape = [self.program.get_constant_size(atom.predicate, 1)]
            constant_term = atom.terms[0]
        else:
            shape = [self.program.get_constant_size(atom.predicate, 0), 1]
            constant_term = atom.terms[1]
        initial_value = get_initial_value_by_name(initializer_name, shape)
        kernel = self._matrix_to_variable_with_value(
            atom, initial_value, shape=shape, constant=constant_term)
        return self._build_simple_fact_layer(atom, kernel)

    def _get_arity_x_x_trainable_iterable_constant_variable(
            self, atom, input_constant=None, input_combining_function=None,
            output_constant=None, output_extract_function=None):
        name = self._get_layer_name(atom)
        variable_atom = get_variable_atom(atom)
        fact_layer = self.build_atom(variable_atom)
        return SpecificFactLayer(
            name, fact_layer,
            input_constant=input_constant,
            input_combining_function=input_combining_function,
            output_constant=output_constant,
            output_extract_function=output_extract_function
        )

    def _get_arity_2_2_not_trainable_constant_variable(self, atom):
        w = self.get_vector_representation_with_constant(atom)
        if isinstance(w, csr_matrix):
            w = w.todense()
        kernel = self._matrix_to_constant(
            atom, w,
            shape=[self.program.get_constant_size(atom.predicate, 1)])
        return self._build_simple_fact_layer(atom, kernel)

    def _get_arity_2_2_not_trainable_variable_constant(self, atom):
        w = self.get_vector_representation_with_constant(atom)
        kernel = self._matrix_to_constant(
            atom, w,
            shape=[self.program.get_constant_size(atom.predicate, 0), 1])
        return self._build_simple_fact_layer(atom, kernel)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(0, 0, False))
    def arity_0_not_trainable(self, atom):
        w = self.get_matrix_representation(atom.predicate)
        kernel = self._matrix_to_constant(atom, w)
        return self._build_simple_fact_layer(atom, kernel)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(0, 0, True))
    def arity_0_trainable(self, atom):
        kernel = self._matrix_to_variable(atom)
        return self._build_simple_fact_layer(atom, kernel)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 0, False, FactoryTermType.NUMBER))
    def arity_1_0_not_trainable_number(self, atom):
        return self._get_arity_x_0_attribute(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 0, True, FactoryTermType.NUMBER))
    def arity_1_0_trainable_number(self, atom):
        return self._get_arity_x_0_attribute(atom, trainable=True)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, False, FactoryTermType.CONSTANT))
    def arity_1_1_not_trainable_constant(self, atom):
        return self._get_not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, False,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_1_1_not_trainable_iterable_constant(self, atom):
        return self._get_not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, False, FactoryTermType.VARIABLE))
    def arity_1_1_not_trainable_variable(self, atom):
        w = self.get_matrix_representation(atom.predicate)
        shape = [self.program.get_constant_size(atom.predicate, 0)]
        kernel = self._matrix_to_constant(atom, w, shape)
        return self._build_simple_fact_layer(atom, kernel)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, True, FactoryTermType.CONSTANT))
    def arity_1_1_trainable_constant(self, atom):
        initial_value = self._get_initial_value_for_atom(atom)
        kernel = self._matrix_to_variable(atom, initial_value)
        return self._build_simple_fact_layer(atom, kernel)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, True,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_1_1_trainable_iterable_constant(self, atom):
        output_constant = self.get_constant_lookup(atom, 0)
        output_extract_func = self.get_output_extract_function(atom.predicate)
        return self._get_arity_x_x_trainable_iterable_constant_variable(
            atom, output_constant=output_constant,
            output_extract_function=output_extract_func)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, True, FactoryTermType.VARIABLE))
    def arity_1_1_trainable_variable(self, atom):
        name = self._get_layer_name(atom)
        predicate = atom.predicate
        initializer_name = self._get_initializer_name(predicate)
        shape = [self.program.get_constant_size(atom.predicate, 0)]
        initial_value = get_initial_value_by_name(initializer_name, shape)
        kernel = self._matrix_to_variable_with_value(atom, initial_value, shape)
        fact_combining_function = self.get_edge_combining_function(predicate)
        return FactLayer(name, kernel, fact_combining_function)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 0, False, FactoryTermType.NUMBER,
                                       FactoryTermType.NUMBER))
    def arity_2_0_not_trainable_number_number(self, atom):
        return self._get_arity_x_0_attribute(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 0, True, FactoryTermType.NUMBER,
                                       FactoryTermType.NUMBER))
    def arity_2_0_trainable_number_number(self, atom):
        return self._get_arity_x_0_attribute(atom, trainable=True)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 1, False, FactoryTermType.CONSTANT,
                                       FactoryTermType.NUMBER))
    def arity_2_1_not_trainable_constant_number(self, atom):
        return self._get_arity_2_1_constant_attribute(atom, attribute_index=1)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 1, False,
                                       FactoryTermType.ITERABLE_CONSTANT,
                                       FactoryTermType.NUMBER))
    def arity_2_1_not_trainable_iterable_constant_number(self, atom):
        return self._get_arity_2_1_iterable_constant_number(atom, 0)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 1, False, FactoryTermType.VARIABLE,
                                       FactoryTermType.NUMBER))
    def arity_2_1_not_trainable_variable_number(self, atom):
        return self._get_arity_2_1_variable_number(atom, 1)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 1, False, FactoryTermType.NUMBER,
                                       FactoryTermType.CONSTANT))
    def arity_2_1_not_trainable_number_constant(self, atom):
        return self._get_arity_2_1_constant_attribute(atom, attribute_index=0)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 1, False,
                                       FactoryTermType.NUMBER,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_2_1_not_trainable_number_iterable_constant(self, atom):
        return self._get_arity_2_1_iterable_constant_number(atom, 1)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 1, False, FactoryTermType.NUMBER,
                                       FactoryTermType.VARIABLE))
    def arity_2_1_not_trainable_number_variable(self, atom):
        return self._get_arity_2_1_variable_number(atom, 0)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 1, True, FactoryTermType.CONSTANT,
                                       FactoryTermType.NUMBER))
    def arity_2_1_trainable_constant_number(self, atom):
        return self._get_arity_2_1_constant_attribute(atom, 1, trainable=True)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 1, True,
                                       FactoryTermType.ITERABLE_CONSTANT,
                                       FactoryTermType.NUMBER))
    def arity_2_1_trainable_iterable_constant_number(self, atom):
        return self._get_arity_2_1_iterable_constant_number(atom, 0)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 1, True, FactoryTermType.VARIABLE,
                                       FactoryTermType.NUMBER))
    def arity_2_1_trainable_variable_number(self, atom):
        return self._get_arity_2_1_variable_number(atom, 1, trainable=True)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 1, True, FactoryTermType.NUMBER,
                                       FactoryTermType.CONSTANT))
    def arity_2_1_trainable_number_constant(self, atom):
        return self._get_arity_2_1_constant_attribute(atom, 0, trainable=True)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 1, True,
                                       FactoryTermType.NUMBER,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_2_1_trainable_number_iterable_constant(self, atom):
        return self._get_arity_2_1_iterable_constant_number(atom, 1)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 1, True, FactoryTermType.NUMBER,
                                       FactoryTermType.VARIABLE))
    def arity_2_1_trainable_number_variable(self, atom):
        return self._get_arity_2_1_variable_number(atom, 0, trainable=True)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False, FactoryTermType.CONSTANT,
                                       FactoryTermType.CONSTANT))
    def arity_2_2_not_trainable_constant_constant(self, atom):
        return self._get_not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       FactoryTermType.CONSTANT,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_2_2_not_trainable_constant_iterable_constant(self, atom):
        return self._get_not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False, FactoryTermType.CONSTANT,
                                       FactoryTermType.VARIABLE))
    def arity_2_2_not_trainable_constant_variable(self, atom):
        return self._get_arity_2_2_not_trainable_constant_variable(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       FactoryTermType.ITERABLE_CONSTANT,
                                       FactoryTermType.CONSTANT))
    def arity_2_2_not_trainable_iterable_constant_constant(self, atom):
        return self._get_not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       FactoryTermType.ITERABLE_CONSTANT,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_2_2_not_trainable_iterable_constant_iterable_constant(self, atom):
        return self._get_not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       FactoryTermType.ITERABLE_CONSTANT,
                                       FactoryTermType.VARIABLE))
    def arity_2_2_not_trainable_iterable_constant_variable(self, atom):
        return self._get_arity_2_2_not_trainable_constant_variable(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False, FactoryTermType.VARIABLE,
                                       FactoryTermType.CONSTANT))
    def arity_2_2_not_trainable_variable_constant(self, atom):
        return self._get_arity_2_2_not_trainable_variable_constant(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       FactoryTermType.VARIABLE,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_2_2_not_trainable_variable_iterable_constant(self, atom):
        return self._get_arity_2_2_not_trainable_variable_constant(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False, FactoryTermType.VARIABLE,
                                       FactoryTermType.VARIABLE))
    def arity_2_2_not_trainable_variable_variable(self, atom):
        if atom.terms[0] == atom.terms[1]:
            # Both terms are equal variables
            w = self.get_diagonal_matrix_representation(atom.predicate)
            shape = [self.program.get_constant_size(atom.predicate, 0)]
            kernel = self._matrix_to_constant(atom, w, shape=shape)
        else:
            w = self.get_matrix_representation(atom.predicate)
            kernel = self._matrix_to_constant(atom, w)
        return self._build_simple_fact_layer(atom, kernel)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True, FactoryTermType.CONSTANT,
                                       FactoryTermType.CONSTANT))
    def arity_2_2_trainable_constant_constant(self, atom):
        initial_value = self._get_initial_value_for_atom(atom)
        kernel = self._matrix_to_variable(atom, initial_value)
        return self._build_simple_fact_layer(atom, kernel)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       FactoryTermType.CONSTANT,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_2_2_trainable_constant_iterable_constant(self, atom):
        variable_atom = Atom(atom.predicate, atom.terms[0], Variable("X"),
                             weight=atom.weight)
        name = self._get_layer_name(atom)
        fact_layer = self.build_atom(variable_atom)
        output_constant = self.get_constant_lookup(atom, 1)
        output_extract_func = self.get_output_extract_function(atom.predicate)
        return SpecificFactLayer(name, fact_layer,
                                 output_constant=output_constant,
                                 output_extract_function=output_extract_func)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True, FactoryTermType.CONSTANT,
                                       FactoryTermType.VARIABLE))
    def arity_2_2_trainable_constant_variable(self, atom):
        return self._get_arity_2_2_trainable_variable_and_constant(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       FactoryTermType.ITERABLE_CONSTANT,
                                       FactoryTermType.CONSTANT))
    def arity_2_2_trainable_iterable_constant_constant(self, atom):
        name = self._get_layer_name(atom)
        variable_atom = Atom(atom.predicate, Variable("X"), atom.terms[1],
                             weight=atom.weight)
        fact_layer = self.build_atom(variable_atom)
        input_constant = self.get_one_hot_tensor(atom, 0)
        input_combining_func = self.get_and_combining_function(atom.predicate)
        return SpecificFactLayer(name, fact_layer,
                                 input_constant=input_constant,
                                 input_combining_function=input_combining_func)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       FactoryTermType.ITERABLE_CONSTANT,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_2_2_trainable_iterable_constant_iterable_constant(self, atom):
        predicate = atom.predicate
        input_constant = self.get_one_hot_tensor(atom, 0)
        input_combining_function = self.get_and_combining_function(predicate)

        output_constant = self.get_constant_lookup(atom, 1)
        output_extract_func = self.get_output_extract_function(atom.predicate)
        return self._get_arity_x_x_trainable_iterable_constant_variable(
            atom, input_constant=input_constant,
            input_combining_function=input_combining_function,
            output_constant=output_constant,
            output_extract_function=output_extract_func)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       FactoryTermType.ITERABLE_CONSTANT,
                                       FactoryTermType.VARIABLE))
    def arity_2_2_trainable_iterable_constant_variable(self, atom):
        predicate = atom.predicate
        input_constant = self.get_one_hot_tensor(atom, 0)
        input_combining_function = self.get_and_combining_function(predicate)
        return self._get_arity_x_x_trainable_iterable_constant_variable(
            atom, input_constant=input_constant,
            input_combining_function=input_combining_function)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True, FactoryTermType.VARIABLE,
                                       FactoryTermType.CONSTANT))
    def arity_2_2_trainable_variable_constant(self, atom):
        return self._get_arity_2_2_trainable_variable_and_constant(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       FactoryTermType.VARIABLE,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_2_2_trainable_variable_iterable_constant(self, atom):
        variable_atom = get_variable_atom(atom)
        name = self._get_layer_name(atom)
        fact_layer = self.build_atom(variable_atom)
        predicate = atom.predicate
        fact_layer = InvertedFactLayer(fact_layer, self, predicate)
        fact_combining_function = self.get_edge_combining_function_2d(predicate)
        output_constant = self.get_one_hot_tensor(atom, 1)
        return InvertedSpecificFactLayer(
            name, fact_layer, fact_combining_function,
            output_constant=output_constant)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True, FactoryTermType.VARIABLE,
                                       FactoryTermType.VARIABLE))
    def arity_2_2_trainable_variable_variable(self, atom):
        predicate = atom.predicate
        if atom.terms[0] == atom.terms[1]:
            variable_atom = get_variable_atom(atom)
            name = self._get_layer_name(atom)
            kernel = self.build_atom(variable_atom).get_kernel()
            fact_combining_func = self.get_edge_combining_function(predicate)
            return DiagonalFactLayer(name, kernel, fact_combining_func)
        else:
            initializer_name = self._get_initializer_name(predicate)
            size_0 = self.program.get_constant_size(atom.predicate, 0)
            size_1 = self.program.get_constant_size(atom.predicate, 1)
            shape = [size_0, size_1]
            initial_value = get_initial_value_by_name(initializer_name,
                                                      shape)
            kernel = self._matrix_to_variable_with_value(
                atom, initial_value, shape=shape)
            return self._build_simple_fact_layer(atom, kernel)
