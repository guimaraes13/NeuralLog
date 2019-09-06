"""
A factory of tensors
"""

import logging
from enum import Enum

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from tensorflow.python.keras import initializers
from typing import Dict, Any

from knowledge.network import NotGroundAtomException
from knowledge.program import NeuralLogProgram
from language.language import Predicate, Atom, Quote, Constant, Variable, Term

logger = logging.getLogger()


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
    for term in atom.terms:
        if term.is_constant():
            if isinstance(term, Quote):
                terms.append(Constant(term.value))
            else:
                terms.append(term)
        else:
            terms.append(Variable("X{}".format(index)))
            index += 1
    return Atom(atom.predicate, *terms, weight=atom.weight)


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
        self.terms = args

    def key(self):
        """
        Specifies the keys to be used in the equals and hash functions.
        :return: the keys
        :rtype: Any
        """
        return self.arity, self.true_arity, self.trainable, tuple(self.terms)

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.key() == other.key()
        return False


class TensorFactory:
    """
    A factory of TensorFlow Tensors.
    """

    SPARSE_THRESHOLD = 0.3

    _tensor_by_atom: Dict[Atom, tf.Tensor] = dict()
    "The tensors for the atoms"

    _tensor_by_constant: Dict[Term, tf.Tensor] = dict()
    "The one-hot tensors for the iterable constants"

    _matrix_representation: Dict[Predicate, Any] = dict()
    "Caches the matrices representations."

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
        term_types = []
        types = self.program.predicates[atom.predicate]
        for i in range(atom.arity()):
            if types[i].number:
                term_types.append(FactoryTermType.NUMBER)
            elif atom.terms[i].is_constant():
                if atom.terms[i] in self.program.iterable_constants:
                    term_types.append(FactoryTermType.ITERABLE_CONSTANT)
                else:
                    term_types.append(FactoryTermType.CONSTANT)
            else:
                term_types.append(FactoryTermType.VARIABLE)

        trainable = atom.predicate in self.program.trainable_predicates
        key = TensorFunctionKey(atom.arity(),
                                self.program.get_true_arity(atom.predicate),
                                trainable, *term_types)
        return self.function[key](atom)

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

    def _matrix_to_variable(self, atom, initial_value, shape=None):
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
            if shape is None:
                if hasattr(initial_value, 'shape'):
                    shape = list(initial_value.shape)
                else:
                    shape = []
            tensor = tf.Variable(initial_value=initial_value, dtype=tf.float32,
                                 shape=shape, name=renamed_atom.__str__())
            # noinspection PyTypeChecker
            self._tensor_by_atom[renamed_atom] = tensor

        return tensor

    def _matrix_to_constant(self, atom, value, shape=None):
        """
        Returns a constant representation of the atom.

        :param atom: the atom
        :type atom: Atom
        :param value: the value of the constant
        :type value: np.array or csr_matrix or float
        :param shape: the shape of the variable
        :type shape: Any
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
            tensor = self._build_constant(renamed_atom, value, shape)
            # noinspection PyTypeChecker
            self._tensor_by_atom[renamed_atom] = tensor

        return tensor

    def _build_constant(self, renamed_atom, value, shape):
        """
        Builds the constant for the atom.

        :param renamed_atom: the atom
        :type renamed_atom: Atom
        :param value: the value of the constant
        :type value: np.array or csr_matrix or float
        :param shape: the shape of the variable
        :type shape: Any
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

        tensor = tf.constant(value=value, dtype=tf.float32,
                             shape=shape, name=renamed_atom.__str__())
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

        initial_value = self.program.facts_by_predicate.get(
            atom.predicate, dict()).get(atom).get(atom.simple_key(), None)
        if initial_value is None:
            initializer_name = "glorot_uniform"
            initializer = initializers.get(initializer_name)
            initializer = initializer()
            initial_value = lambda: initializer([], dtype=tf.float32)
            logger.debug("Creating atom %s with initializer %s",
                         atom, initializer_name)
        else:
            initial_value = initial_value.weight

        return initial_value

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
                                name=constant.value)
            self._tensor_by_constant[constant] = tensor

        return tensor

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(0, 0, False, []))
    def arity_0_not_trainable(self, atom):
        w = self.get_matrix_representation(atom.predicate)
        return self._matrix_to_constant(atom, w)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(0, 0, True, []))
    def arity_0_trainable(self, atom):
        w = self.get_matrix_representation(atom.predicate)
        return self._matrix_to_variable(atom, w)

    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(1, 0, False, [FactoryTermType.NUMBER]))
    # def arity_1_0_not_trainable_number(self, atom):
    #     pass
    #
    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(1, 0, True, [FactoryTermType.NUMBER]))
    # def arity_1_0_trainable_number(self, atom):
    #     pass

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, False, [FactoryTermType.CONSTANT]))
    def arity_1_1_not_trainable_constant(self, atom):
        return self._not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, False,
                                       [FactoryTermType.ITERABLE_CONSTANT]))
    def arity_1_1_not_trainable_iterable_constant(self, atom):
        return self._not_trainable_constants(atom)

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

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, False, [FactoryTermType.VARIABLE]))
    def arity_1_1_not_trainable_variable(self, atom):
        w = self.get_matrix_representation(atom.predicate)
        return self._matrix_to_constant(atom, w, [self.constant_size, 1])

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, True, [FactoryTermType.CONSTANT]))
    def arity_1_1_trainable_constant(self, atom):
        initial_value = self._get_initial_value_for_atom(atom)
        return self._matrix_to_variable(atom, initial_value)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, True,
                                       [FactoryTermType.ITERABLE_CONSTANT]))
    def arity_1_1_trainable_iterable_constant(self, atom):
        vector = self.get_matrix_representation(atom.predicate)
        variable_atom = Atom(atom.predicate, Variable("X"), weight=atom.weight)
        tensor = self._matrix_to_variable(variable_atom, vector.todense(),
                                          [self.constant_size, 1])
        index = self._get_one_hot_tensor(atom.terms[0])
        return tf.matmul(index, tensor, a_is_sparse=True)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(1, 1, True, [FactoryTermType.VARIABLE]))
    def arity_1_1_trainable_variable(self, atom):
        w = self.get_matrix_representation(atom.predicate)
        return self._matrix_to_variable(atom, w.todense(),
                                        [self.constant_size, 1])

    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(2, 0, False, [FactoryTermType.NUMBER,
    #                                                  FactoryTermType.NUMBER]))
    # def arity_2_0_not_trainable_number_number(self, atom):
    #     pass
    #
    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(2, 0, True, [FactoryTermType.NUMBER,
    #                                                 FactoryTermType.NUMBER]))
    # def arity_2_0_trainable_number_number(self, atom):
    #     pass
    #
    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(2, 1, False, [FactoryTermType.CONSTANT,
    #                                                  FactoryTermType.NUMBER]))
    # def arity_2_1_not_trainable_constant_number(self, atom):
    #     pass
    #
    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(2, 1, False,
    #                                    [FactoryTermType.ITERABLE_CONSTANT,
    #                                     FactoryTermType.NUMBER]))
    # def arity_2_1_not_trainable_iterable_constant_number(self, atom):
    #     pass
    #
    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(2, 1, False, [FactoryTermType.VARIABLE,
    #                                                  FactoryTermType.NUMBER]))
    # def arity_2_1_not_trainable_variable_number(self, atom):
    #     pass
    #
    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(2, 1, True, [FactoryTermType.CONSTANT,
    #                                                 FactoryTermType.NUMBER]))
    # def arity_2_1_trainable_constant_number(self, atom):
    #     pass
    #
    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(2, 1, True,
    #                                    [FactoryTermType.ITERABLE_CONSTANT,
    #                                     FactoryTermType.NUMBER]))
    # def arity_2_1_trainable_iterable_constant_number(self, atom):
    #     pass
    #
    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(2, 1, True, [FactoryTermType.VARIABLE,
    #                                                 FactoryTermType.NUMBER]))
    # def arity_2_1_trainable_variable_number(self, atom):
    #     pass
    #
    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(2, 1, False, [FactoryTermType.NUMBER,
    #                                                  FactoryTermType.CONSTANT]))
    # def arity_2_1_not_trainable_number_constant(self, atom):
    #     pass
    #
    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(2, 1, False,
    #                                    [FactoryTermType.NUMBER,
    #                                     FactoryTermType.ITERABLE_CONSTANT]))
    # def arity_2_1_not_trainable_number_iterable_constant(self, atom):
    #     pass
    #
    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(2, 1, False, [FactoryTermType.NUMBER,
    #                                                  FactoryTermType.VARIABLE]))
    # def arity_2_1_not_trainable_number_variable(self, atom):
    #     pass
    #
    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(2, 1, True, [FactoryTermType.NUMBER,
    #                                                 FactoryTermType.CONSTANT]))
    # def arity_2_1_trainable_number_constant(self, atom):
    #     pass
    #
    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(2, 1, True,
    #                                    [FactoryTermType.NUMBER,
    #                                     FactoryTermType.ITERABLE_CONSTANT]))
    # def arity_2_1_trainable_number_iterable_constant(self, atom):
    #     pass
    #
    # # noinspection PyMissingOrEmptyDocstring
    # @tensor_function(TensorFunctionKey(2, 1, True, [FactoryTermType.NUMBER,
    #                                                 FactoryTermType.VARIABLE]))
    # def arity_2_1_trainable_number_variable(self, atom):
    #     pass

    # noinspection PyMissingOrEmptyDocstring

    @tensor_function(TensorFunctionKey(2, 2, False, [FactoryTermType.CONSTANT,
                                                     FactoryTermType.CONSTANT]))
    def arity_2_2_not_trainable_constant_constant(self, atom):
        return self._not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       [FactoryTermType.CONSTANT,
                                        FactoryTermType.ITERABLE_CONSTANT]))
    def arity_2_2_not_trainable_constant_iterable_constant(self, atom):
        return self._not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    def _not_trainable_constant_variable(self, atom):
        w = self.program.get_vector_representation_with_constant(atom)
        return self._matrix_to_constant(atom, w)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False, [FactoryTermType.CONSTANT,
                                                     FactoryTermType.VARIABLE]))
    def arity_2_2_not_trainable_constant_variable(self, atom):
        return self._not_trainable_constant_variable(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       [FactoryTermType.ITERABLE_CONSTANT,
                                        FactoryTermType.CONSTANT]))
    def arity_2_2_not_trainable_iterable_constant_constant(self, atom):
        return self._not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       [FactoryTermType.ITERABLE_CONSTANT,
                                        FactoryTermType.ITERABLE_CONSTANT]))
    def arity_2_2_not_trainable_iterable_constant_iterable_constant(self, atom):
        return self._not_trainable_constants(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       [FactoryTermType.ITERABLE_CONSTANT,
                                        FactoryTermType.VARIABLE]))
    def arity_2_2_not_trainable_iterable_constant_variable(self, atom):
        return self._not_trainable_constant_variable(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False, [FactoryTermType.VARIABLE,
                                                     FactoryTermType.CONSTANT]))
    def arity_2_2_not_trainable_variable_constant(self, atom):
        return self._not_trainable_constant_variable(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False,
                                       [FactoryTermType.VARIABLE,
                                        FactoryTermType.ITERABLE_CONSTANT]))
    def arity_2_2_not_trainable_variable_iterable_constant(self, atom):
        return self._not_trainable_constant_variable(atom)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, False, [FactoryTermType.VARIABLE,
                                                     FactoryTermType.VARIABLE]))
    def arity_2_2_not_trainable_variable_variable(self, atom):
        if atom.terms[0] == atom.terms[1]:
            # Both terms are equal variables
            w = self.get_diagonal_matrix_representation(atom.predicate)
            return self._matrix_to_constant(atom, w)
        else:
            w = self.get_matrix_representation(atom.predicate)
            return self._matrix_to_constant(atom, w)

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True, [FactoryTermType.CONSTANT,
                                                     FactoryTermType.CONSTANT]))
    def arity_2_2_trainable_constant_constant(self, atom):
        # TODO: implement
        pass

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       [FactoryTermType.CONSTANT,
                                        FactoryTermType.ITERABLE_CONSTANT]))
    def arity_2_2_trainable_constant_iterable_constant(self, atom):
        # TODO: implement
        pass

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True, [FactoryTermType.CONSTANT,
                                                     FactoryTermType.VARIABLE]))
    def arity_2_2_trainable_constant_variable(self, atom):
        # TODO: implement
        pass

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       [FactoryTermType.ITERABLE_CONSTANT,
                                        FactoryTermType.CONSTANT]))
    def arity_2_2_trainable_iterable_constant_constant(self, atom):
        # TODO: implement
        pass

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       [FactoryTermType.ITERABLE_CONSTANT,
                                        FactoryTermType.ITERABLE_CONSTANT]))
    def arity_2_2_trainable_iterable_constant_iterable_constant(self, atom):
        # TODO: implement
        pass

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       [FactoryTermType.ITERABLE_CONSTANT,
                                        FactoryTermType.VARIABLE]))
    def arity_2_2_trainable_iterable_constant_variable(self, atom):
        # TODO: implement
        pass

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True, [FactoryTermType.VARIABLE,
                                                     FactoryTermType.CONSTANT]))
    def arity_2_2_variable_constant(self, atom):
        # TODO: implement
        pass

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True,
                                       [FactoryTermType.VARIABLE,
                                        FactoryTermType.ITERABLE_CONSTANT]))
    def arity_2_2_variable_iterable_constant(self, atom):
        # TODO: implement
        pass

    # noinspection PyMissingOrEmptyDocstring
    @tensor_function(TensorFunctionKey(2, 2, True, [FactoryTermType.VARIABLE,
                                                     FactoryTermType.VARIABLE]))
    def arity_2_2_variable_variable(self, atom):
        w = self.get_matrix_representation(atom.predicate)
        tensor = self._matrix_to_variable(atom, w.todense())
        if atom.terms[0] == atom.terms[1]:
            tensor = tf.linalg.tensor_diag_part(tensor)
            tensor = tf.reshape(tensor, [-1, 1])
        return tensor
