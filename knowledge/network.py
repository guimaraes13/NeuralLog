"""
Compiles the language into a neural network.
"""
import logging

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from tensorflow import keras
from tensorflow.python.keras import initializers
from typing import Dict, Any

from knowledge.program import NeuralLogProgram
from knowledge.tensor_factory import TensorFactory
from language.language import Predicate, Atom, Variable, Quote, Constant, Term

# Network part
# TODO: create the neural network representation
# TODO: build the rule layer and its computation, based on the path
# TODO: find the paths in the target rule
# TODO: build the predicates presented in the body of the target rules
# TODO: recursively build the predicates needed
# TODO: connect the rule layers to create the network
# TODO: create a function to transform the examples from logic to numeric
# TODO: create a function to extracted the weights learned by the network
# TODO: treaty the binary atom with equal variables

logger = logging.getLogger()


class NotGroundAtomException(Exception):
    """
    Represents an atom malformed exception.
    """

    def __init__(self, atom) -> None:
        """
        Creates a not ground atom exception.

        :param atom: the not ground atom
        :type Atom: Atom
        """
        super().__init__("Atom {} is not ground.".format(atom))


class PredicateLayer(keras.layers.Layer):
    """
    A Layer to combine the inputs of a predicate. The inputs of a predicate
    are the facts of this predicate and the result of rules with this
    predicate in their heads.

    The default combining function for these inputs is the sum.
    """

    def __init__(self, name, combining_function, **kwargs):
        """
        Creates a PredicateLayer.

        :param predicate: the predicate of the layer
        :type predicate: Predicate
        :param combining_function: the combining function.
        :type combining_function: function
        :param kwargs: additional arguments
        :type kwargs: dict
        """
        self.combining_function = combining_function
        kwargs["name"] = name
        super(PredicateLayer, self).__init__(**kwargs)

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        return self.combining_function(inputs)

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(PredicateLayer, self).get_config()

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class RuleLayer(keras.layers.Layer):
    """
    A Layer to represent a logic rule.
    """

    def __init__(self, **kwargs):
        super(RuleLayer, self).__init__(**kwargs)

    # noinspection PyMissingOrEmptyDocstring
    def build(self, input_shape):
        pass

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        pass

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        pass

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        pass

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class FactLayer(keras.layers.Layer):
    """
    A Layer to represent the logic facts from a predicate.
    """

    def __init__(self, **kwargs):
        super(FactLayer, self).__init__(**kwargs)

    # noinspection PyMissingOrEmptyDocstring
    def build(self, input_shape):
        pass

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        pass

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        pass

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        pass

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NeuralLogNetwork(keras.Model):
    """
    The NeuralLog Network.
    """

    SPARSE_THRESHOLD = 0.3

    _layer_by_predicate: Dict[Predicate, PredicateLayer] = dict()
    "The layer by predicate"

    _tensor_by_atom: Dict[Atom, tf.Tensor] = dict()
    "The tensors for the atoms"

    _tensor_by_constant: Dict[Term, tf.Tensor] = dict()
    "The one-hot tensors for the iterable constants"

    _matrix_representation: Dict[Predicate, Any] = dict()
    "Caches the matrices representations."

    _diagonal_matrix_representation: Dict[Predicate, csr_matrix] = dict()
    "Caches the diagonal matrices representations."

    def __init__(self, program, predicate_combining=tf.math.accumulate_n):
        """
        Creates a NeuralLogNetwork.

        :param program: the neural language
        :type program: NeuralLogProgram
        :param predicate_combining: the combining function. The default
        combining function is `sum`, implemented by `tf.accumulate_n`
        :type predicate_combining: function
        """
        super(NeuralLogNetwork, self).__init__(name="NeuralLogNetwork")
        self.program = program
        self.constant_size = len(self.program.iterable_constants)
        self.predicate_combining = predicate_combining
        self.path_combining = None
        self.edge_combining = None
        self.tensor_factory = TensorFactory(self.program)

    # TODO: For each target predicate, start a queue with the predicate
    # TODO: For each predicate in the queue, find the rules for the predicate
    # TODO:
    def build_network(self):
        """
        Builds the neural network based on the language and the set of examples.
        """
        for predicate in self.program.examples:
            self._build_predicate(Atom(predicate,
                                       *list(map(lambda x: "X{}".format(x),
                                                 range(predicate.arity)))))

    def _build_predicate(self, atom):
        """
        Builds the layer for the atom.

        :param atom: the atom
        :type atom: Atom
        :return: the predicate layer
        :rtype: Predicate
        """
        inputs = [self._build_fact(atom)]
        for clause in self.program.clauses_by_predicate.get(atom.predicate, []):
            rule = self._build_rule(clause)
            if rule is not None:
                inputs.append(rule)
        predicate_layer = PredicateLayer(atom, self.predicate_combining)
        # return predicate_layer(inputs)

    def _build_fact(self, atom):
        """
        Builds the fact layer for the atom.

        :param atom: the atom
        :type atom: Atom
        :return: the fact layer
        :rtype: FactLayer
        """

        return None

    def _build_rule(self, clause):
        return None

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

    def get_matrix_representation_for_atom(self, atom):
        # TODO: get the tensor representation for the literal instead of the
        #  predicate:
        #  - adjust for literals with constants;
        #  - adjust for literals with numeric attributes;
        #  - adjust for negated literals;
        """
        Builds the matrix representation for the atom.

        :param atom: the atom
        :type atom: Atom
        :raise UnsupportedMatrixRepresentation in the case the predicate is
        not convertible to matrix form
        :return: the matrix representation of the data for the given predicate
        :rtype: csr_matrix or np.matrix or (csr_matrix, csr_matrix) or float
        """
        if atom.arity() == 0:
            w = self.get_matrix_representation(atom.predicate)
            if atom.predicate in self.program.trainable_predicates:
                return self._matrix_to_variable(atom, w)
            else:
                return self._matrix_to_constant(atom, w)

        if atom.arity() == 1:
            if atom.terms[0].is_constant():
                # Unary atom with constant
                constant = atom.terms[0]
                if atom.predicate in self.program.trainable_predicates:
                    # The predicate is trainable
                    if constant in self.program.iterable_constants:
                        # The constant is iterable, get the vector for the
                        # predicate and multiply by the index of the constant
                        vector = self.get_matrix_representation(
                            atom.predicate)
                        variable_atom = Atom(atom.predicate,
                                             Variable("X"),
                                             weight=atom.weight)
                        tensor = self._matrix_to_variable(
                            variable_atom, vector.todense(),
                            [self.constant_size, 1])
                        index = self._get_one_hot_tensor(constant)
                        return tf.matmul(index, tensor, a_is_sparse=True)
                    else:
                        # The constant is not iterable, get the weight of the
                        # fact, if it does not exists, create one
                        initial_value = self._get_initial_value_for_atom(atom)
                        return self._matrix_to_variable(atom, initial_value)
                else:
                    # The predicate is not trainable
                    # Return the constant of the fact directly, if it does not
                    # exist, send a warning and assign 0.0 weight to it
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
            else:
                w = self.get_matrix_representation(atom.predicate)
                if atom.predicate in self.program.trainable_predicates:
                    return self._matrix_to_variable(
                        atom, w.todense(), [self.constant_size, 1])
                else:
                    return self._matrix_to_constant(atom, w,
                                                    [self.constant_size, 1])

        # TODO: make share that all variable weights of a fact point to the
        #  same tensor variable
        if atom.arity() == 2:
            if atom.terms[0].is_constant() and atom.terms[1].is_constant():
                # Both terms are constants
                # TODO: implement, binary atom with two constants
                if atom.predicate in self.program.trainable_predicates:
                    # Trainable
                    if atom.terms[0] in self.program.iterable_constants and \
                            atom.terms[1] in self.program.iterable_constants:
                        pass
                    elif atom.terms[0] in self.program.iterable_constants:
                        pass
                    elif atom.terms[1] in self.program.iterable_constants:
                        pass
                    else:
                        pass
                else:
                    # Not trainable
                    pass
            elif atom.terms[0].is_constant():
                # The first terms is constant and the second is variable
                # TODO: implement, binary atom with first term constants
                pass
            elif atom.terms[1].is_constant():
                # The first terms is variable and the second is constant
                # TODO: implement, binary atom with second term constants
                pass
            else:
                # Both terms are variables
                if atom.predicate in self.program.trainable_predicates:
                    w = self.get_matrix_representation(atom.predicate)
                    tensor = self._matrix_to_variable(atom, w.todense())
                    if atom.terms[0] == atom.terms[1]:
                        tensor = tf.linalg.tensor_diag_part(tensor)
                        tensor = tf.reshape(tensor, [-1, 1])
                    return tensor
                else:
                    if atom.terms[0] == atom.terms[1]:
                        # Both terms are equal variables
                        w = self.get_diagonal_matrix_representation(
                            atom.predicate)
                        return self._matrix_to_constant(atom, w)
                    else:
                        w = self.get_matrix_representation(atom.predicate)
                        return self._matrix_to_constant(atom, w)

        return self.tensor_factory.build_atom(atom)

    def _build_two_variables_tensor(self, atom, value):
        """
        Builds a tensor for the atom with two variables.

        :param atom: the atom with two variables
        :type atom: Atom
        :param value: the initial value
        :type value: csr_matrix, np.matrix
        :return: the tensor for the atom
        :rtype: tf.Tensor or tf.SparseTensor
        """
        if atom.predicate in self.program.trainable_predicates:
            return self._matrix_to_variable(atom, value)
        else:
            return self._matrix_to_constant(atom, value)

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


if __name__ == "__main__":
    # x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    # y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
    #
    # linear_model = tf.layers.Dense(units=1)
    #
    # y_pred = linear_model(x)
    # loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    #
    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    # train = optimizer.minimize(loss)
    #
    # init = tf.global_variables_initializer()
    #
    # sess = tf.Session()
    # sess.run(init)
    # for i in range(1000):
    #     _, loss_value = sess.run((train, loss))
    #     # print(loss_value)
    #
    # print(sess.run(y_pred))

    # a = tf.constant(3.0, dtype=tf.float32)
    # a = tf.SparseTensor(indices=[[0, 0], [1, 2]],
    #                     values=[1.0, 2.0], dense_shape=[3, 4])
    a = tf.SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [0, 3],
                                 [1, 0], [1, 1], [1, 2], [1, 3],
                                 [2, 0], [2, 1], [2, 2], [2, 3],
                                 ],
                        values=[2.0, 3.0, 5.0, 7.0,
                                11.0, 13.0, 17.0, 19.0,
                                23.0, 19.0, 31.0, 37.0], dense_shape=[4, 4])
    # a = tf.Variable(4.0)

    # b=tf.constant([[1.0], [0.0], [1.0], [0.0]]) # also tf.float32 implicitly

    b = tf.constant([[2.0, 3.0, 5.0, 7.0]], name="B")  # also tf.float32
    # implicitly
    total = tf.sparse.sparse_dense_matmul(a, b, adjoint_a=True, adjoint_b=True)
    # a = tf.sparse.to_dense(a)
    total = tf.transpose(total, name="Total")
    # b = tf.SparseTensor(indices=[[0, 1], [0, 2]], values=[2, 1],
    #                     dense_shape=[2, 7])
    # w = tf.SparseTensor(indices=[[0, 1], [1, 2]], values=[1, 1],
    #                     dense_shape=[2, 7])
    # total = tf.nn.embedding_lookup_sparse(a, b, sp_weights=w, combiner="sum")
    print(a)
    print(b)
    print(total)

    sess = tf.compat.v1.Session()
    # writer = tf.compat.v1.summary.FileWriter("/Users/Victor/Desktop",
    #                                          sess.graph)
    # init = tf.compat.v1.global_variables_initializer()
    # # tf.sparse_add
    # sess.run(init)
    # # _a, _t = sess.run((tf.sparse.to_dense(a), total))
    # # _a, _b, _t = sess.run((a, tf.sparse.to_dense(b), total))
    # _a, _b, _t = sess.run((tf.sparse.to_dense(a), b, total))
    # print(_a)
    # print()
    # print(_b)
    # print()
    # print(_t)
    # writer.close()

    indices = tf.constant([[0, 0], [1, 1]], dtype=tf.int64)
    values = tf.constant([1, 1])
    dynamic_input = tf.compat.v1.placeholder(tf.float32, shape=[None, None])
    s = tf.shape(dynamic_input, out_type=tf.int64)

    st = tf.SparseTensor(indices, values, s)
    st_ordered = tf.sparse.reorder(st)
    result = tf.sparse.to_dense(st_ordered)

    _r = sess.run(result, feed_dict={dynamic_input: np.zeros([5, 3])})
    print(_r)
    print()

    _r = sess.run(result, feed_dict={dynamic_input: np.zeros([3, 3])})
    print(_r)
    print()

    # a = tf.constant([[0, 1, 0]])
    # b = tf.constant([[2, 3], [5, 7], [11, 13]])
    # c = tf.matmul(a, b)
    # _c = sess.run(c)
    # print(_c)
    # print("\n")
    _a = sess.run(a)
    print(_a)
    print("\n")
    _da = sess.run(tf.sparse.to_dense(a))
    print(_da)

    print("\n")
    part = tf.linalg.tensor_diag_part(tf.sparse.to_dense(a))
    part = tf.reshape(part, [-1, 1])
    _dda = sess.run(part)
    print(_dda)
    print(_dda.shape)

    # s = np.array([[1.0, 0.0, 0.0, 0.0],
    #               [1.0, 0.0, 2.0, 0.0],
    #               [0.0, 0.0, 0.0, 0.0],
    #               [0.0, 0.0, 0.0, 1.0]])
    # sparse = csr_matrix(s)
    #
    # print(sparse)
    # # v_sparse = tf.constant(sparse)
    # v_sparse = tf.constant(sparse.todense())
    # # init = tf.global_variables_initializer()
    # # sess.run(init)
    #
    # result = sess.run(v_sparse)
    # print(result)
