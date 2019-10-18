"""
Compiles the language into a neural network.
"""
import logging
from collections import deque
from typing import Dict, Set, List

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from tensorflow.python import keras

from src.knowledge.tensor_factory import TensorFactory, get_standardised_name
from src.language.language import HornClause, Literal
from src.language.language import Predicate, Atom, Term

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

# IMPROVE: test if the values of trainable iterable constants and trainable
#  variables are pointing to the same variable.
#  Skip this to the network test.
ANY_PREDICATE = "any"

logger = logging.getLogger()


def is_cyclic(atom, previous_atoms):
    """
    Check if there is a cycle between the current atom and the previous
    atoms. If the atom's predicate appears in a atom in previous atoms,
    then, there is a cycle.

    :param atom: the current atom
    :type atom: Atom
    :param previous_atoms: the previous atoms
    :type previous_atoms: list[Atom] or set[Atom]
    :return: True if there is a cycle; False, otherwise
    :rtype: bool
    """
    if previous_atoms is None or len(previous_atoms) == 0:
        return False

    for previous_atom in previous_atoms:
        if atom.predicate == previous_atom.predicate:
            return True

    return False


def get_disconnected_literals(clause, connected_literals):
    """
    Gets the literals from the `clause` which are disconnected from the
    source and destination variables but are grounded.

    :param clause: the clause
    :type clause: HornClause
    :param connected_literals: the set of connected literals
    :type connected_literals: Set[Literal]
    :return: the list of disconnected literals
    :rtype: List[Literal]
    """
    ground_literals = []
    for literal in clause.body:
        if literal in connected_literals or not literal.is_grounded():
            continue
        ground_literals.append(literal)
    return ground_literals


def build_literal_dictionaries(clause):
    """
    Builds the dictionaries with the literals of the `clause`. This method
    creates two dictionaries, the first one containing the literals that
    connects different terms; and the second one the loop literals,
    which connects the terms to themselves, either by being unary or by having
    equal terms.

    Both dictionaries have the terms in the literals as keys.

    :param clause: the clause
    :type clause: HornClause
    :return: the dictionaries
    :rtype: (Dict[Term, List[Literal]], Dict[Term, List[Literal]])
    """
    binary_literals_by_term = dict()  # type: Dict[Term, List[Literal]]
    loop_literals_by_term = dict()  # type: Dict[Term, List[Literal]]
    for literal in clause.body:
        if literal.arity() == 1:
            dict_to_append = loop_literals_by_term
        elif literal.arity() == 2:
            if literal.terms[0] == literal.terms[-1]:
                dict_to_append = loop_literals_by_term
            else:
                dict_to_append = binary_literals_by_term
        else:
            continue
        for term in set(literal.terms):
            dict_to_append.setdefault(term, []).append(literal)
    return binary_literals_by_term, loop_literals_by_term


def find_paths(partial_paths, destination, binary_literals_by_term,
               completed_paths, visited_literals):
    """
    Finds the paths from `partial_paths` to `destination` by appending the
    literals from `binary_literals_by_term`. The completed paths are stored
    in `completer_paths` while the used literals are stores in
    `visited_literals`.

    Finally, it returns the dead end paths.

    :param partial_paths: The initial partial paths
    :type partial_paths: deque[RulePath]
    :param destination: the destination term
    :type destination: Term
    :param binary_literals_by_term: the literals to be appended
    :type binary_literals_by_term: Dict[Term, List[Literals]]
    :param completed_paths: the completed paths
    :type completed_paths: deque[RulePath]
    :param visited_literals: the visited literals
    :type visited_literals: Set[Literal]
    :return: the dead end paths
    :rtype: List[RulePath]
    """
    dead_end_paths = []  # type: List[RulePath]
    while len(partial_paths) > 0:
        size = len(partial_paths)
        for i in range(size):
            path = partial_paths.popleft()
            path_end = path.path_end()

            if path_end == destination:
                completed_paths.append(path)
                continue

            possible_edges = binary_literals_by_term.get(path_end, [None])
            not_added_path = True
            for literal in possible_edges:
                new_path = path.new_path_with_item(literal)
                if new_path is None:
                    continue
                partial_paths.append(new_path)
                # noinspection PyTypeChecker
                visited_literals.add(literal)
                not_added_path = False

            if not_added_path:
                dead_end_paths.append(path)
    return dead_end_paths


def append_not_in_set(literal, literals, appended):
    """
    Appends `literal` to `literals` if it is not in `appended`.
    Also, updates appended.

    :param literal: the literal to append
    :type literal: Literal
    :param literals: the list to append to
    :type literals: List[Literal]
    :param appended: the set of already appended literals
    :type appended: Set[Literal]
    """
    if literal not in appended:
        literals.append(literal)
        appended.add(literal)


def complete_path_with_any(dead_end_paths, destination,
                           inverted=False):
    """
    Completes the path by appending the special `any` predicate between the
    end of the path and the destination.

    :param dead_end_paths: the paths to be completed
    :type dead_end_paths: collections.Iterable[RulePath]
    :param destination: the destination
    :type destination: Term
    :param inverted: if `True`, append the any predicate with the terms in the
    reversed order
    :type inverted: bool
    :return: the completed paths
    :rtype: List[RulePath]
    """
    completed_paths = []
    for path in dead_end_paths:
        if inverted:
            any_literal = Literal(Atom(ANY_PREDICATE,
                                       destination, path.path_end()))
        else:
            any_literal = Literal(Atom(ANY_PREDICATE,
                                       path.path_end(), destination))
        path.append(any_literal)
        completed_paths.append(path)
    return completed_paths


def append_loop_predicates(completed_paths, loop_literals_by_term,
                           visited_literals, reverse_path=False):
    """
    Appends the loop predicates to the path.

    :param completed_paths: the completed paths
    :type completed_paths: deque[RulePath]
    :param loop_literals_by_term: the loop predicates by term
    :type loop_literals_by_term: Dict[Term, List[Literals]]
    :param visited_literals: the set of visited literals to be updated
    :type visited_literals: Set[Literal]
    :param reverse_path: if `True`, reverse the path before processing
    :type reverse_path: bool
    :return: the final paths
    :rtype: List[RulePath]
    """
    final_paths = []  # type: List[RulePath]
    for path in completed_paths:
        if reverse_path:
            path = path.reverse()
        last_reversed = False
        literals = []
        appended = set()
        for i in range(len(path.path)):
            last_reversed = path.inverted[i]
            input_term = path[i].terms[-1 if last_reversed else 0]
            output_term = path[i].terms[0 if last_reversed else -1]
            for literal in loop_literals_by_term.get(input_term, []):
                append_not_in_set(literal, literals, appended)
            literals.append(path.path[i])
            for literal in loop_literals_by_term.get(output_term, []):
                append_not_in_set(literal, literals, appended)
                last_reversed = False
            visited_literals.update(appended)
        final_paths.append(RulePath(literals, last_reversed))
    return final_paths


def find_all_forward_paths(source, destination,
                           loop_literals_by_term,
                           binary_literals_by_term, visited_literals):
    """
    Finds all forward paths from `source` to `destination` by using the
    literals in `binary_literals_by_term` and `loop_literals_by_term`.
    If the destination cannot be reached, it includes a special `any`
    predicated to connect the path.

    :param source: the source of the paths
    :type source: Term
    :param destination: the destination of the paths
    :type destination: Term
    :param loop_literals_by_term: the literals that connects different terms
    :type loop_literals_by_term: Dict[Term, List[Literals]]
    :param binary_literals_by_term: the loop literals, which connects the
    terms to themselves
    :type binary_literals_by_term: Dict[Term, List[Literals]]
    :param visited_literals: the set of visited literals
    :type visited_literals: Set[Literal]
    :return: the completed forward paths between source and destination
    :rtype: List[RulePath]
    """
    partial_paths = deque()  # type: deque[RulePath]
    completed_paths = deque()  # type: deque[RulePath]

    for literal in binary_literals_by_term.get(source, []):
        inverted = literal.terms[-1] == source
        partial_paths.append(RulePath([literal], inverted))
        visited_literals.add(literal)

    dead_end_paths = find_paths(
        partial_paths, destination, binary_literals_by_term,
        completed_paths, visited_literals)

    for path in complete_path_with_any(
            dead_end_paths, destination, inverted=False):
        completed_paths.append(path)

    return append_loop_predicates(
        completed_paths, loop_literals_by_term, visited_literals,
        reverse_path=False)


def find_all_backward_paths(source, destination,
                            loop_literals_by_term,
                            binary_literals_by_term, visited_literals):
    """
    Finds all backward paths from `source` to `destination` by using the
    literals in `binary_literals_by_term` and `loop_literals_by_term`.
    If the destination cannot be reached, it includes a special `any`
    predicated to connect the path.

    :param source: the source of the paths
    :type source: Term
    :param destination: the destination of the paths
    :type destination: Term
    :param loop_literals_by_term: the literals that connects different terms
    :type loop_literals_by_term: Dict[Term, List[Literals]]
    :param binary_literals_by_term: the loop literals, which connects the
    terms to themselves
    :type binary_literals_by_term: Dict[Term, List[Literals]]
    :param visited_literals: the set of visited literals
    :type visited_literals: Set[Literal]
    :return: the completed backward paths between source and destination
    :rtype: List[RulePath]
    """
    partial_paths = deque()  # type: deque[RulePath]
    completed_paths = deque()  # type: deque[RulePath]

    for literal in binary_literals_by_term.get(source, []):
        if literal in visited_literals:
            continue
        inverted = literal.terms[-1] == source
        partial_paths.append(RulePath([literal], inverted))
        visited_literals.add(literal)

    dead_end_paths = find_paths(
        partial_paths, destination, binary_literals_by_term,
        completed_paths, visited_literals)

    for path in complete_path_with_any(
            dead_end_paths, destination, inverted=True):
        completed_paths.append(path)

    return append_loop_predicates(
        completed_paths, loop_literals_by_term, visited_literals,
        reverse_path=True)


def find_clause_paths(clause):
    """
    Finds the paths in the clause.
    :param clause: the clause
    :type clause: HornClause
    """
    # Defining variables
    binary_literals_by_term, loop_literals_by_term = \
        build_literal_dictionaries(clause)
    visited_literals = set()

    # Finding forward paths
    source = clause.head.terms[0]
    destination = clause.head.terms[-1]
    forward_paths = find_all_forward_paths(
        source, destination, loop_literals_by_term,
        binary_literals_by_term, visited_literals)

    # Finding backward paths
    source = clause.head.terms[-1]
    destination = clause.head.terms[0]
    backward_paths = find_all_backward_paths(
        source, destination, loop_literals_by_term,
        binary_literals_by_term, visited_literals)

    ground_literals = get_disconnected_literals(
        clause, visited_literals)

    return forward_paths + backward_paths, ground_literals


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

        :param name: the name of the layer
        :type name: str
        :param combining_function: the combining function.
        :type combining_function: function
        :param kwargs: additional arguments
        :type kwargs: dict
        """
        # noinspection PyTypeChecker
        kwargs["name"] = name
        super(PredicateLayer, self).__init__(**kwargs)
        self.combining_function = combining_function

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


class FactLayer(keras.layers.Layer):
    """
    A Layer to represent the logic facts from a predicate.
    """

    def __init__(self, name, kernel, combining_function, **kwargs):
        """
        Creates a PredicateLayer.

        :param name: the name of the layer
        :type name: str
        :param kernel: the data of the layer.
        :type kernel: tf.Tensor
        :param combining_function: the combining function.
        :type combining_function: function
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        # noinspection PyTypeChecker
        kwargs["name"] = name
        super(FactLayer, self).__init__(**kwargs)
        self.kernel = kernel
        self.combining_function = combining_function

    # noinspection PyMissingOrEmptyDocstring
    def build(self, input_shape):
        return tf.TensorShape(input_shape)

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        return self.combining_function(inputs, self.kernel)

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(FactLayer, self).get_config()

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


class RulePath:
    """
    Represents a rule path.
    """

    path: List[Literal] = list()
    "The path of literals"

    literals: Set[Literal] = set()
    "The set of literals in the path"

    terms: Set[Term]
    "The set of all terms in the path"

    inverted: List[bool]
    """It is True if the correspondent literal is inverted;
    it is false, otherwise"""

    def __init__(self, path, last_inverted=False):
        """
        Initializes a path.

        :param path: the path
        :type path: collections.Iterable[Literal]
        :param last_inverted: if the last literal is inverted
        :type last_inverted: bool
        """
        self.path = list(path)
        self.literals = set(path)
        # self.last_term = self.path[-1].terms[0 if inverted else -1]
        self.terms = set()
        for literal in self.literals:
            self.terms.update(literal.terms)
        self.inverted = self._compute_inverted(last_inverted)

    def _compute_inverted(self, last_inverted):
        inverted = []
        last_term = self.path[-1].terms[0 if last_inverted else -1]
        for i in reversed(range(0, len(self.path))):
            literal = self.path[i]
            if literal.arity() != 1 and literal.terms[0] == last_term:
                inverted.append(True)
                last_term = literal.terms[-1]
            else:
                inverted.append(False)
                last_term = literal.terms[0]

        return list(reversed(inverted))

    def __getitem__(self, item):
        return self.path.__getitem__(item)

    def append(self, item):
        """
        Appends the item to the end of the path, if it is not in the path yet.

        :param item: the item
        :type item: Literal
        :return: True, if the item has been appended to the path; False,
        otherwise.
        :rtype: bool
        """
        if item is None:
            return False

        if item.arity() == 1:
            output_variable = item.terms[0]
            last_inverted = False
        else:
            if item.terms[0] == self.path_end():
                output_variable = item.terms[-1]
                last_inverted = False
            else:
                output_variable = item.terms[0]
                last_inverted = True

        if item.arity() != 1 and output_variable in self.terms:
            return False

        self.path.append(item)
        self.literals.add(item)
        self.terms.update(item.terms)
        self.inverted.append(last_inverted)
        return True

    def new_path_with_item(self, item):
        """
        Creates a new path with `item` at the end.

        :param item: the item
        :type item: Literal or None
        :return: the new path, if its is possible to append the item; None,
        otherwise
        :rtype: RulePath or None
        """
        path = RulePath(self.path)
        return path if path.append(item) else None

    def path_end(self):
        """
        Gets the term at the end of the path.

        :return: the term at the end of the path
        :rtype: Term
        """
        return self.path[-1].terms[0 if self.inverted[-1] else -1]

    def __str__(self):
        message = ""
        for i in reversed(range(0, len(self.path))):
            literal = self.path[i]
            prefix = literal.predicate.name
            iterator = list(map(lambda x: x.value, literal.terms))
            if self.inverted[i]:
                prefix += "^{-1}"
                iterator = reversed(iterator)

            prefix += "("
            prefix += ", ".join(iterator)
            prefix += ")"
            message = prefix + message
            if i > 0:
                message = ", " + message

        return message

    def reverse(self):
        """
        Gets a reverse path.

        :return: the reverse path
        :rtype: RulePath
        """
        not_inverted = self.path[0].arity() != 1 and not self.inverted[0]
        path = RulePath(reversed(self.path), not_inverted)
        return path

    __repr__ = __str__


class NeuralLogNetwork(keras.Model):
    """
    The NeuralLog Network.
    """

    _layer_by_predicate: Dict[Predicate, PredicateLayer] = dict()
    "The layer by predicate"

    # TODO: Create a way to define the functions in the program
    def __init__(self, program,
                 predicate_combining_function=tf.math.accumulate_n,
                 path_combining_function=tf.math.multiply,
                 edge_combining_function=tf.tensordot):
        """
        Creates a NeuralLogNetwork.

        :param program: the neural language
        :type program: NeuralLogProgram
        :param predicate_combining_function: the combining function. The default
        combining function is `sum`, implemented by `tf.accumulate_n`
        :type predicate_combining_function: function
        """
        super(NeuralLogNetwork, self).__init__(name="NeuralLogNetwork")
        self.program = program
        self.constant_size = len(self.program.iterable_constants)
        self.predicate_combining_function = predicate_combining_function
        self.path_combining_function = path_combining_function
        self.edge_combining_function = edge_combining_function
        self.tensor_factory = TensorFactory(self.program)
        self.queue = deque()

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

    # noinspection PyDefaultArgument
    def _build_predicate(self, atom, previous_atoms=[]):
        """
        Builds the layer for the atom.

        :param atom: the atom
        :type atom: Atom
        :return: the predicate layer
        :rtype: PredicateLayer
        """
        if is_cyclic(atom, previous_atoms):
            logger.error("Error: cannot create the Predicate Node for %s", atom)
            raise Exception()
        inputs = [self._build_fact(atom)]
        for clause in self.program.clauses_by_predicate.get(atom.predicate, []):
            rule = self._build_rule(clause, previous_atoms)
            if rule is not None:
                inputs.append(rule)
        predicate_layer = PredicateLayer(
            "pred_layer_{}".format(get_standardised_name(atom.__str__())),
            self.predicate_combining_function)
        return predicate_layer(inputs)

    def _build_rule(self, clause, previous_atoms=[]):
        """
        Builds the Rule Node.
        :param clause: the clause
        :type clause: HornClause
        """
        current_atoms = previous_atoms + [clause.head]
        predicate_nodes = []
        for literal in clause.body:
            predicate_nodes.append(self._build_predicate(literal,
                                                         current_atoms))
            # TODO: Create the paths and join them
        return None

    def _build_fact(self, atom):
        """
        Builds the fact layer for the atom.

        :param atom: the atom
        :type atom: Atom
        :return: the fact layer
        :rtype: FactLayer
        """
        return FactLayer(
            "fact_layer_{}".format(get_standardised_name(atom.__str__())),
            self.tensor_factory.build_atom(atom), self.edge_combining_function)

    def get_matrix_representation_for_atom(self, atom):
        # TODO: get the tensor representation for the literal instead of the
        #  predicate:
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
        return self.tensor_factory.build_atom(atom)

# if __name__ == "__main__":
#     # x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
#     # y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
#     #
#     # linear_model = tf.layers.Dense(units=1)
#     #
#     # y_pred = linear_model(x)
#     # loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
#     #
#     # optimizer = tf.train.GradientDescentOptimizer(0.1)
#     # train = optimizer.minimize(loss)
#     #
#     # init = tf.global_variables_initializer()
#     #
#     # sess = tf.Session()
#     # sess.run(init)
#     # for i in range(1000):
#     #     _, loss_value = sess.run((train, loss))
#     #     # print(loss_value)
#     #
#     # print(sess.run(y_pred))
#
#     # a = tf.constant(3.0, dtype=tf.float32)
#     # a = tf.SparseTensor(indices=[[0, 0], [1, 2]],
#     #                     values=[1.0, 2.0], dense_shape=[3, 4])
#     a = tf.SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [0, 3],
#                                  [1, 0], [1, 1], [1, 2], [1, 3],
#                                  [2, 0], [2, 1], [2, 2], [2, 3],
#                                  ],
#                         values=[2.0, 3.0, 5.0, 7.0,
#                                 11.0, 13.0, 17.0, 19.0,
#                                 23.0, 19.0, 31.0, 37.0], dense_shape=[4, 4])
#     # a = tf.Variable(4.0)
#
#     # b=tf.constant([[1.0], [0.0], [1.0], [0.0]]) # also tf.float32 implicitly
#
#     b = tf.constant([[2.0, 3.0, 5.0, 7.0]], name="B")  # also tf.float32
#     # implicitly
#     total = tf.sparse.sparse_dense_matmul(a, b,
#     adjoint_a=True, adjoint_b=True)
#     # a = tf.sparse.to_dense(a)
#     total = tf.transpose(total, name="Total")
#     # b = tf.SparseTensor(indices=[[0, 1], [0, 2]], values=[2, 1],
#     #                     dense_shape=[2, 7])
#     # w = tf.SparseTensor(indices=[[0, 1], [1, 2]], values=[1, 1],
#     #                     dense_shape=[2, 7])
#     # total = tf.nn.embedding_lookup_sparse(a, b,
#     sp_weights=w, combiner="sum")
#     print(a)
#     print(b)
#     print(total)
#
#     sess = tf.compat.v1.Session()
#     # writer = tf.compat.v1.summary.FileWriter("/Users/Victor/Desktop",
#     #                                          sess.graph)
#     # init = tf.compat.v1.global_variables_initializer()
#     # # tf.sparse_add
#     # sess.run(init)
#     # # _a, _t = sess.run((tf.sparse.to_dense(a), total))
#     # # _a, _b, _t = sess.run((a, tf.sparse.to_dense(b), total))
#     # _a, _b, _t = sess.run((tf.sparse.to_dense(a), b, total))
#     # print(_a)
#     # print()
#     # print(_b)
#     # print()
#     # print(_t)
#     # writer.close()
#
#     indices = tf.constant([[0, 0], [1, 1]], dtype=tf.int64)
#     values = tf.constant([1, 1])
#     dynamic_input = tf.compat.v1.placeholder(tf.float32, shape=[None, None])
#     s = tf.shape(dynamic_input, out_type=tf.int64)
#
#     st = tf.SparseTensor(indices, values, s)
#     st_ordered = tf.sparse.reorder(st)
#     result = tf.sparse.to_dense(st_ordered)
#
#     _r = sess.run(result, feed_dict={dynamic_input: np.zeros([5, 3])})
#     print(_r)
#     print()
#
#     _r = sess.run(result, feed_dict={dynamic_input: np.zeros([3, 3])})
#     print(_r)
#     print()
#
#     # a = tf.constant([[0, 1, 0]])
#     # b = tf.constant([[2, 3], [5, 7], [11, 13]])
#     # c = tf.matmul(a, b)
#     # _c = sess.run(c)
#     # print(_c)
#     # print("\n")
#     _a = sess.run(a)
#     print(_a)
#     print("\n")
#     _da = sess.run(tf.sparse.to_dense(a))
#     print(_da)
#
#     print("\n")
#     part = tf.linalg.tensor_diag_part(tf.sparse.to_dense(a))
#     part = tf.reshape(part, [-1, 1])
#     _dda = sess.run(part)
#     print(_dda)
#     print(_dda.shape)
#
#     # s = np.array([[1.0, 0.0, 0.0, 0.0],
#     #               [1.0, 0.0, 2.0, 0.0],
#     #               [0.0, 0.0, 0.0, 0.0],
#     #               [0.0, 0.0, 0.0, 1.0]])
#     # sparse = csr_matrix(s)
#     #
#     # print(sparse)
#     # # v_sparse = tf.constant(sparse)
#     # v_sparse = tf.constant(sparse.todense())
#     # # init = tf.global_variables_initializer()
#     # # sess.run(init)
#     #
#     # result = sess.run(v_sparse)
#     # print(result)
