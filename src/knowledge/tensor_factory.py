"""
A factory of tensors
"""

import logging
from enum import Enum

import numpy as np
import re
import tensorflow as tf
from scipy.sparse import csr_matrix
from tensorflow.python.keras import initializers
from typing import Dict, Any

from src.language.language import Predicate, Atom, Quote, Constant, Variable, \
    Term

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


def get_renamed_atom(atom):
    """
    Gets a renamed atom, replacing their variables for a positional name.
    In this way, atoms with different variable names will have the same key,
    as long as the number of variables and their positions matches.

    :param atom: the atom
    :type atom: Atom
    :return: the renamed atom
    :rtype: Atom
    """
    terms = []
    index = 0
    term_map = dict()
    for term in atom.terms:
        if term.is_constant():
            if isinstance(term, Quote):
                terms.append(Constant(term.value))
            else:
                terms.append(term)
        else:
            if term not in term_map:
                term_map[term] = "X{}".format(index)
                index += 1
            terms.append(term_map[term])
    return Atom(atom.predicate, *terms)


def get_variable_atom(atom):
    """
    Gets an atom by replacing their constant for unique variables.

    :param atom: the atom
    :type atom: Atom
    :return: the renamed atom
    :rtype: Atom
    """
    terms = [Variable("X{}".format(i)) for i in range(atom.arity())]
    return Atom(atom.predicate, *terms, weight=atom.weight)


def get_initial_value_by_name(initializer, shape, **kwargs):
    """
    Gets the initializer by name or config.

    :param initializer: the initializer name or config
    :type initializer: str, dict
    :param shape: the shape of the initial value
    :type shape: int or collections.Iterable[int]
    :return: the initial value
    :rtype: function
    """
    initializer = initializers.get(initializer)
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
    if FIRST_CHARACTER.search(standard[0]) is not None:
        return "." + standard

    return standard


class TensorFactory:
    """
    A factory of TensorFlow Tensors.
    """

    SPARSE_THRESHOLD = 0.3

    _tensor_by_atom: Dict[Atom, tf.Tensor] = dict()
    "The tensors for the atoms"

    _tensor_by_name: Dict[str, tf.Tensor] = dict()
    "The tensors by name"

    _tensor_by_constant: Dict[Term, tf.Tensor] = dict()
    "The one-hot tensors for the iterable constants"

    _matrix_representation: Dict[Predicate, Any] = dict()
    "Caches the matrices representations."

    _vector_representation: Dict[Atom, Any] = dict()
    "Caches the vector representations of binary predicates with constant."

    _diagonal_matrix_representation: Dict[Predicate, csr_matrix] = dict()
    "Caches the diagonal matrices representations."

    tensor_function = decorate_factory_function()

    def __init__(self, program):
        """
        Creates a TensorFactory.

        :param program: the NeuralLog program
        :type program: NeuralLogProgram
        """
        self.program = program
        self.constant_size = len(self.program.iterable_constants)
        # noinspection PyUnresolvedReferences
        self.function = self.tensor_function.functions
        self.weight_attribute_combining_function = tf.multiply
        self.attribute_attribute_combining_function = tf.multiply

    def build_atom(self, atom):
        """
        Builds the matrix representation for the atom.

        :param atom: the atom
        :type atom: Atom
        :raise UnsupportedMatrixRepresentation in the case the predicate is
        not convertible to matrix form
        :return: the matrix representation of the data for the given predicate
        :rtype: csr_matrix or np.matrix or (csr_matrix, csr_matrix) or float
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
                if atom.terms[i] in self.program.iterable_constants.inverse:
                    term_types.append(FactoryTermType.ITERABLE_CONSTANT)
                else:
                    term_types.append(FactoryTermType.CONSTANT)
            else:
                term_types.append(FactoryTermType.VARIABLE)
        trainable = atom.predicate in self.program.trainable_predicates
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

    def _get_variable(self, name, value, shape, dtype=tf.float32):
        tensor = self._tensor_by_name.get(name, None)
        if tensor is None:
            tensor = tf.Variable(initial_value=value, dtype=dtype,
                                 shape=shape, name=get_standardised_name(name))
        return tensor

    def _get_constant(self, name, value, shape, dtype=tf.float32):
        tensor = self._tensor_by_name.get(name, None)
        if tensor is None:
            tensor = tf.constant(value=value, dtype=dtype,
                                 shape=shape, name=get_standardised_name(name))
        return tensor

    def _matrix_to_variable(self, atom, initial_value=None, shape=None):
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
        tensor = self._tensor_by_atom.get(renamed_atom, None)
        if tensor is None:
            if initial_value is None:
                initial_value = self.get_matrix_representation(atom.predicate)
            if shape is None:
                if hasattr(initial_value, 'shape'):
                    shape = list(initial_value.shape)
                else:
                    shape = []
            if isinstance(initial_value, csr_matrix):
                initial_value = initial_value.todense()

            tensor = self._get_variable(renamed_atom.__str__(),
                                        initial_value, shape)
            # noinspection PyTypeChecker
            self._tensor_by_atom[renamed_atom] = tensor

        return tensor

    def _matrix_to_variable_with_value(self, atom, initial_value, shape=None):
        """
        Returns a variable representation of the atom, initializing the
        unknown values with initial value.

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
        tensor = self._tensor_by_atom.get(renamed_atom, None)
        if tensor is None:
            if shape is None:
                if hasattr(initial_value, 'shape'):
                    shape = list(initial_value.shape)
                else:
                    shape = []
            tensor = self._get_variable(
                "init-{}".format(renamed_atom.__str__()), initial_value, shape)
            weights = self.get_matrix_representation(atom.predicate).todense()
            # IMPROVE: create a function to get a mask with 1 where the fact
            #  is defined or 0 otherwise.
            mask = self.program.get_matrix_representation(
                atom.predicate, mask=True)
            if isinstance(mask, csr_matrix):
                mask = mask.todense()
            mask = 1 - mask
            # noinspection PyTypeChecker
            mask = self._build_constant(
                mask, shape, "mask-{}".format(renamed_atom.__str__()))
            tensor = tf.multiply(tensor, mask)
            w = self._matrix_to_constant(atom, weights, shape)
            tensor = tf.add(tensor, w)
            # noinspection PyTypeChecker
            self._tensor_by_atom[renamed_atom] = tensor

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
        tensor = self._tensor_by_atom.get(renamed_atom, None)
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
            tensor = self._build_constant(value, shape, name)
            # noinspection PyTypeChecker
            self._tensor_by_atom[renamed_atom] = tensor

        return tensor

    def _build_constant(self, value, shape, name):
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
        if isinstance(value, csr_matrix):
            sparsity = len(value.data) / np.prod(value.shape, dtype=np.float32)
            if sparsity < self.SPARSE_THRESHOLD:
                data = value.data
                rows, columns = value.nonzero()
                tensor = tf.SparseTensor(
                    indices=list(map(lambda x: list(x), zip(rows, columns))),
                    values=data, dense_shape=value.shape)
                return tf.sparse.reorder(tensor)
            else:
                value = value.todense()

        return self._get_constant(name, value, shape)

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
        predicate_parameter = self.program.parameters.get(
            predicate, self.program.parameters)
        initializer_name = predicate_parameter["initial_value"]

        return initializer_name

    def _get_one_hot_tensor(self, constant):
        """
        Gets an one-hot row tensor for the iterable constant.

        :param constant: the iterable constant
        :type constant: Term
        :return: the one-hot row tensor
        :rtype: tf.Tensor
        """
        tensor = self._tensor_by_constant.get(constant, None)
        if tensor is None:
            tensor = tf.one_hot([self.program.index_for_constant(constant)],
                                depth=self.constant_size, dtype=tf.float32,
                                name=get_standardised_name(constant.value))
            self._tensor_by_constant[constant] = tensor

        return tensor

    # noinspection PyMissingOrEmptyDocstring
    def _not_trainable_constants(self, atom):
        fact = self.program.facts_by_predicate.get(
            atom.predicate, dict()).get(atom.simple_key(), None)
        if fact is None:
            weight = 0.0
            if atom.context is not None:
                logger.warning(
                    "Warning: there is no fact matching the atom "
                    "%s at line %d:%d, weight replaced by %d.",
                    atom, atom.context.start.line,
                    atom.context.start.column, weight)
            else:
                logger.warning(
                    "Warning: there is no fact matching the atom "
                    "%s, weight replaced by %d.", atom, weight)
        else:
            weight = fact.weight
        return self._matrix_to_constant(atom, weight)

    def _get_arity_x_0_attribute(self, atom, trainable=False):
        tensor = self._tensor_by_atom.get(atom, None)
        if tensor is None:
            weight, value = self.get_matrix_representation(atom.predicate)
            for i in range(atom.arity()):
                if atom.terms[i].is_constant():
                    if value[i] != atom.terms[i].value:
                        weight = 0.0
            renamed_atom = get_renamed_atom(atom)
            if trainable:
                w_tensor = self._get_variable("w-{}".format(
                    renamed_atom.__str__()), weight, [])
            else:
                w_tensor = self._get_constant("w-{}".format(
                    renamed_atom.__str__()), weight, [])
            v_tensor = self._get_constant("v-{}-0".format(
                renamed_atom.__str__()), value[0], [])
            for i in range(1, atom.arity()):
                v_tensor_i = self._get_constant(
                    "v-{}-{}".format(renamed_atom.__str__(), i), value[i], [])
                v_tensor = self.attribute_attribute_combining_function(
                    v_tensor, v_tensor_i)
            tensor = self.weight_attribute_combining_function(
                w_tensor, v_tensor,
                name=get_standardised_name(renamed_atom.__str__()))
            self._tensor_by_atom[atom] = tensor
        return tensor

    def _get_arity_2_1_constant_attribute(self, atom, attribute_index,
                                          trainable=False):
        renamed_atom = get_renamed_atom(atom)
        tensor = self._tensor_by_atom.get(renamed_atom, None)
        if tensor is None:
            atom_value = self.program.facts_by_predicate.get(
                atom.predicate, dict()).get(atom.simple_key(), None)
            if atom_value is None:
                weight = 0.0
                value = 0.0
                logger.warning(
                    "Warning: there is no fact matching the atom "
                    "%s, weight replaced by %d.", atom, weight)
            elif not atom.terms[attribute_index].is_constant() or \
                    atom.terms[attribute_index] == \
                    atom_value.terms[attribute_index]:
                weight = atom_value.weight
                value = atom_value.terms[attribute_index].value
            else:
                weight = 0.0
                value = atom.terms[attribute_index].value
            if trainable:
                w_tensor = self._get_variable("w-{}".format(
                    renamed_atom.__str__()), weight, [])
            else:
                w_tensor = self._get_constant(
                    "w-{}".format(renamed_atom.__str__()), weight, [])
            v_tensor = self._get_constant(
                "v-{}".format(renamed_atom.__str__()), value, [])
            tensor = self.weight_attribute_combining_function(
                w_tensor, v_tensor,
                name=get_standardised_name(renamed_atom.__str__()))

            self._tensor_by_atom[renamed_atom] = tensor
        return tensor

    # noinspection PyMissingOrEmptyDocstring
    def _get_weights_and_values(self, atom, trainable=False,
                                constant_value=None):
        weight, value = self.get_matrix_representation(atom.predicate)
        if constant_value is not None:
            weight = weight.copy()
            for i in range(len(weight.data)):
                if value.data[i] != constant_value:
                    weight.data[i] = 0.0
        renamed_atom = get_renamed_atom(atom)
        w_tensor = self._get_constant("w:{}".format(renamed_atom.__str__()),
                                      weight.todense(), [self.constant_size, 1])
        if trainable:
            v_tensor = self._get_variable("v-{}".format(renamed_atom.__str__()),
                                          value.todense(),
                                          [self.constant_size, 1])
        else:
            v_tensor = self._get_constant("v:{}".format(renamed_atom.__str__()),
                                          value.todense(),
                                          [self.constant_size, 1])
        return v_tensor, w_tensor

    def _get_arity_2_1_iterable_constant_number(self, atom, constant_index,
                                                trainable=False):
        renamed_atom = get_renamed_atom(atom)
        tensor = self._tensor_by_atom.get(renamed_atom, None)
        if tensor is None:
            v_tensor, w_tensor = self._get_weights_and_values(
                atom, trainable)
            index = self._get_one_hot_tensor(atom.terms[constant_index])
            weight = tf.linalg.matmul(index, w_tensor, a_is_sparse=True)
            value = tf.linalg.matmul(index, v_tensor, a_is_sparse=True)
            tensor = self.weight_attribute_combining_function(weight, value)
            tensor = tf.reshape(tensor, [])
            self._tensor_by_atom[renamed_atom] = tensor
        return tensor

    def _get_arity_2_1_variable_number(self, atom, attribute_index,
                                       trainable=False):
        renamed_atom = get_renamed_atom(atom)
        tensor = self._tensor_by_atom.get(renamed_atom, None)
        if tensor is None:
            if atom.terms[attribute_index].is_constant():
                weight, value = self._get_weights_and_values(
                    atom, trainable,
                    constant_value=atom.terms[attribute_index].value)
            else:
                weight, value = self._get_weights_and_values(atom, trainable)
            tensor = self.weight_attribute_combining_function(weight, value)

            self._tensor_by_atom[renamed_atom] = tensor

        return tensor

    # noinspection PyMissingOrEmptyDocstring
    def _not_trainable_constant_variable(self, atom):
        w = self.get_vector_representation_with_constant(atom)
        return self._matrix_to_constant(atom, w)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(0, 0, False))
    def arity_0_not_trainable(self, atom):
        w = self.get_matrix_representation(atom.predicate)
        return self._matrix_to_constant(atom, w)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(0, 0, True))
    def arity_0_trainable(self, atom):
        return self._matrix_to_variable(atom)

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
        return self._not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, False,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_1_1_not_trainable_iterable_constant(self, atom):
        return self._not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, False, FactoryTermType.VARIABLE))
    def arity_1_1_not_trainable_variable(self, atom):
        w = self.get_matrix_representation(atom.predicate)
        return self._matrix_to_constant(atom, w, [self.constant_size, 1])

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, True, FactoryTermType.CONSTANT))
    def arity_1_1_trainable_constant(self, atom):
        initial_value = self._get_initial_value_for_atom(atom)
        return self._matrix_to_variable(atom, initial_value)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, True,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_1_1_trainable_iterable_constant(self, atom):
        tensor = self._tensor_by_atom.get(atom, None)
        if tensor is None:
            variable_atom = get_variable_atom(atom)
            tensor = self.build_atom(variable_atom)
            index = self._get_one_hot_tensor(atom.terms[0])
            tensor = tf.linalg.matmul(index, tensor, a_is_sparse=True)
            tensor = tf.reshape(tensor, [])
            self._tensor_by_atom[atom] = tensor
        return tensor

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, True, FactoryTermType.VARIABLE))
    def arity_1_1_trainable_variable(self, atom):
        initializer_name = self._get_initializer_name(atom.predicate)
        shape = [self.constant_size, 1]
        initial_value = get_initial_value_by_name(initializer_name, shape)
        tensor = self._matrix_to_variable_with_value(atom, initial_value, shape)
        return tensor

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
        return self._get_arity_2_1_iterable_constant_number(atom, 0,
                                                            trainable=True)

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
        return self._get_arity_2_1_iterable_constant_number(atom, 1,
                                                            trainable=True)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 1, True, FactoryTermType.NUMBER,
                                       FactoryTermType.VARIABLE))
    def arity_2_1_trainable_number_variable(self, atom):
        return self._get_arity_2_1_variable_number(atom, 0, trainable=True)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False, FactoryTermType.CONSTANT,
                                       FactoryTermType.CONSTANT))
    def arity_2_2_not_trainable_constant_constant(self, atom):
        return self._not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       FactoryTermType.CONSTANT,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_2_2_not_trainable_constant_iterable_constant(self, atom):
        return self._not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False, FactoryTermType.CONSTANT,
                                       FactoryTermType.VARIABLE))
    def arity_2_2_not_trainable_constant_variable(self, atom):
        return self._not_trainable_constant_variable(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       FactoryTermType.ITERABLE_CONSTANT,
                                       FactoryTermType.CONSTANT))
    def arity_2_2_not_trainable_iterable_constant_constant(self, atom):
        return self._not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       FactoryTermType.ITERABLE_CONSTANT,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_2_2_not_trainable_iterable_constant_iterable_constant(self, atom):
        return self._not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       FactoryTermType.ITERABLE_CONSTANT,
                                       FactoryTermType.VARIABLE))
    def arity_2_2_not_trainable_iterable_constant_variable(self, atom):
        return self._not_trainable_constant_variable(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False, FactoryTermType.VARIABLE,
                                       FactoryTermType.CONSTANT))
    def arity_2_2_not_trainable_variable_constant(self, atom):
        return self._not_trainable_constant_variable(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       FactoryTermType.VARIABLE,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_2_2_not_trainable_variable_iterable_constant(self, atom):
        return self._not_trainable_constant_variable(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False, FactoryTermType.VARIABLE,
                                       FactoryTermType.VARIABLE))
    def arity_2_2_not_trainable_variable_variable(self, atom):
        if atom.terms[0] == atom.terms[1]:
            # Both terms are equal variables
            w = self.get_diagonal_matrix_representation(atom.predicate)
            return self._matrix_to_constant(atom, w)
        else:
            w = self.get_matrix_representation(atom.predicate)
            return self._matrix_to_constant(atom, w)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True, FactoryTermType.CONSTANT,
                                       FactoryTermType.CONSTANT))
    def arity_2_2_trainable_constant_constant(self, atom):
        initial_value = self._get_initial_value_for_atom(atom)
        return self._matrix_to_variable(atom, initial_value)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       FactoryTermType.CONSTANT,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_2_2_trainable_constant_iterable_constant(self, atom):
        tensor = self._tensor_by_atom.get(atom, None)
        if tensor is None:
            variable_atom = Atom(atom.predicate, atom.terms[0], Variable("X"),
                                 weight=atom.weight)
            tensor = self.build_atom(variable_atom)
            index = self._get_one_hot_tensor(atom.terms[1])
            tensor = tf.linalg.matmul(index, tensor, a_is_sparse=True)
            tensor = tf.reshape(tensor, [])
            self._tensor_by_atom[atom] = tensor

        return tensor

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True, FactoryTermType.CONSTANT,
                                       FactoryTermType.VARIABLE))
    def arity_2_2_trainable_constant_variable(self, atom):
        weights = self.get_vector_representation_with_constant(atom)
        return self._matrix_to_variable(atom, weights)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       FactoryTermType.ITERABLE_CONSTANT,
                                       FactoryTermType.CONSTANT))
    def arity_2_2_trainable_iterable_constant_constant(self, atom):
        tensor = self._tensor_by_atom.get(atom, None)
        if tensor is None:
            variable_atom = Atom(atom.predicate, Variable("X"), atom.terms[1],
                                 weight=atom.weight)
            tensor = self.build_atom(variable_atom)
            index = self._get_one_hot_tensor(atom.terms[0])
            tensor = tf.linalg.matmul(tensor, index, b_is_sparse=True,
                                      transpose_a=True, transpose_b=True)
            tensor = tf.reshape(tensor, [])
            self._tensor_by_atom[atom] = tensor

        return tensor

    # noinspection PyMissingOrEmptyDocstring, DuplicatedCode
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       FactoryTermType.ITERABLE_CONSTANT,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_2_2_trainable_iterable_constant_iterable_constant(self, atom):
        tensor = self._tensor_by_atom.get(atom, None)
        if tensor is None:
            variable_atom = get_variable_atom(atom)
            tensor = self.build_atom(variable_atom)
            index_0 = self._get_one_hot_tensor(atom.terms[0])
            index_1 = self._get_one_hot_tensor(atom.terms[1])
            index_1 = tf.reshape(index_1, [-1, 1])
            tensor = tf.linalg.matmul(index_0, tensor, a_is_sparse=True)
            tensor = tf.linalg.matmul(tensor, index_1, b_is_sparse=True)
            tensor = tf.reshape(tensor, [])
            self._tensor_by_atom[atom] = tensor
        return tensor

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       FactoryTermType.ITERABLE_CONSTANT,
                                       FactoryTermType.VARIABLE))
    def arity_2_2_trainable_iterable_constant_variable(self, atom):
        renamed_atom = get_renamed_atom(atom)
        tensor = self._tensor_by_atom.get(renamed_atom, None)
        if tensor is None:
            variable_atom = get_variable_atom(atom)
            tensor = self.build_atom(variable_atom)
            index = self._get_one_hot_tensor(atom.terms[0])
            tensor = tf.linalg.matmul(index, tensor, a_is_sparse=True)
            tensor = tf.reshape(tensor, [-1, 1])
            self._tensor_by_atom[renamed_atom] = tensor
        return tensor

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True, FactoryTermType.VARIABLE,
                                       FactoryTermType.CONSTANT))
    def arity_2_2_trainable_variable_constant(self, atom):
        weights = self.get_vector_representation_with_constant(atom)
        return self._matrix_to_variable(atom, weights)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       FactoryTermType.VARIABLE,
                                       FactoryTermType.ITERABLE_CONSTANT))
    def arity_2_2_trainable_variable_iterable_constant(self, atom):
        renamed_atom = get_renamed_atom(atom)
        tensor = self._tensor_by_atom.get(renamed_atom, None)
        if tensor is None:
            variable_atom = get_variable_atom(atom)
            tensor = self.build_atom(variable_atom)
            index = self._get_one_hot_tensor(atom.terms[1])
            index = tf.reshape(index, [-1, 1])
            tensor = tf.linalg.matmul(tensor, index, b_is_sparse=True)
            self._tensor_by_atom[renamed_atom] = tensor
        return tensor

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True, FactoryTermType.VARIABLE,
                                       FactoryTermType.VARIABLE))
    def arity_2_2_trainable_variable_variable(self, atom):
        if atom.terms[0] == atom.terms[1]:
            renamed_atom = get_renamed_atom(atom)
            tensor = self._tensor_by_atom.get(renamed_atom, None)
            if tensor is None:
                tensor = self._matrix_to_variable(
                    Atom(atom.predicate, "X0", "X1"))
                tensor = tf.linalg.tensor_diag_part(tensor)
                tensor = tf.reshape(tensor, [-1, 1])
                self._tensor_by_atom[renamed_atom] = tensor
            return tensor

        return self._matrix_to_variable(atom)
