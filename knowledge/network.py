"""
Compiles the language into a neural network.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from knowledge.program import NeuralLogProgram


# Network part
# TODO: create the neural network representation
# TODO: build the rule layer and its computation, based on the path
# TODO: find the paths in the target rule
# TODO: build the predicates presented in the body of the target rules
# TODO: recursively build the predicates needed
# TODO: connect the rule layers to create the network
# TODO: create a function to transform the examples from logic to numeric
# TODO: create a function to extracted the weights learned by the network

class PredicateLayer(keras.layers.Layer):
    """
    A Layer to combine the inputs of a predicate. The inputs of a predicate
    are the facts of this predicate and the result of rules with this
    predicate in their heads.

    The default combining function for these inputs is the sum.
    """

    def __init__(self, predicate, combining_function=tf.math.accumulate_n,
                 **kwargs):
        """
        Creates a PredicateLayer.

        :param predicate: the predicate of the layer
        :type predicate: Predicate
        :param combining_function: the combining function. The default
        combining function is `sum`, implemented by `tf.accumulate_n`
        :type combining_function: function
        :param kwargs: additional arguments
        :type kwargs: dict
        """
        self.predicate = predicate
        self.combining_function = combining_function
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

    def __init__(self, program):
        """
        Creates a NeuralLogNetwork.

        :param program: the neural language
        :type program: NeuralLogProgram
        """
        super(NeuralLogNetwork, self).__init__(name="NeuralLogNetwork")
        self.program = program
        self.head_combining = None
        self.path_combining = None
        self.edge_combining = None

    # TODO: For each target predicate, start a queue with the predicate
    # TODO: For each predicate in the queue, find the rules for the predicate
    # TODO:
    def build_network(self):
        """
        Builds the neural network based on the language and the set of examples.
        """
        for predicate in self.program.examples:
            self._build_predicate(predicate)

    def _build_predicate(self, predicate):
        """
        Builds the layer for the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        """
        # fact = self._build_fact(predicate)
        # for rule in self.language.cla


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
                                23.0, 19.0, 31.0, 37.0], dense_shape=[3, 4])
    # a = tf.Variable(4.0)

    # b=tf.constant([[1.0], [0.0], [1.0], [0.0]]) # also tf.float32 implicitly

    b = tf.constant([[2.0, 3.0, 5.0]], name="B")  # also tf.float32 implicitly
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
    dynamic_input = tf.placeholder(tf.float32, shape=[None, None])
    s = tf.shape(dynamic_input, out_type=tf.int64)

    st = tf.SparseTensor(indices, values, s)
    st_ordered = tf.sparse_reorder(st)
    result = tf.sparse_tensor_to_dense(st_ordered)

    _r = sess.run(result, feed_dict={dynamic_input: np.zeros([5, 3])})
    print(_r)
    print()

    _r = sess.run(result, feed_dict={dynamic_input: np.zeros([3, 3])})
    print(_r)
    print()
