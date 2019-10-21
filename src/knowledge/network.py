"""
Compiles the language into a neural network.
"""
import logging
from collections import deque
from typing import Dict, Set, List, Tuple

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from tensorflow.python import keras

from src.knowledge.tensor_factory import TensorFactory, \
    get_standardised_name
from src.language.language import Atom, Term, HornClause, Literal, \
    get_renamed_literal, get_substitution

# Network part
# TODO: create the neural network representation
# TODO: create a function to transform the examples from logic to numeric
# TODO: create a function to extracted the weights learned by the network
# TODO: test everything

# IMPROVE: test if the values of trainable iterable constants and trainable
#  variables are pointing to the same variable.
#  Skip this to the network test.

# WARNING: Do not support literals with same variable in the head of rules.
# WARNING: Do not support literals with constant numbers in the rules.
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
            if get_substitution(previous_atom, atom) is not None:
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


def find_clause_paths(clause, inverted=False):
    """
    Finds the paths in the clause.
    :param inverted: if `True`, creates the paths for the inverted rule;
    this is, the rule in the format (output, input). If `False`,
    creates the path for the standard (input, output) rule format.
    :type inverted: bool
    :param clause: the clause
    :type clause: HornClause
    :return: the completed paths between the terms of the clause and the
    remaining grounded literals
    :rtype: (List[RulePath], List[Literal])
    """
    # Defining variables
    source = clause.head.terms[0]
    destination = clause.head.terms[-1]
    if inverted:
        source, destination = destination, source
    binary_literals_by_term, loop_literals_by_term = \
        build_literal_dictionaries(clause)
    visited_literals = set()

    # Finding forward paths
    forward_paths = find_all_forward_paths(
        source, destination, loop_literals_by_term,
        binary_literals_by_term, visited_literals)

    # Finding backward paths
    source, destination = destination, source
    backward_paths = find_all_backward_paths(
        source, destination, loop_literals_by_term,
        binary_literals_by_term, visited_literals)

    ground_literals = get_disconnected_literals(
        clause, visited_literals)

    return forward_paths + backward_paths, ground_literals


class CyclicProgramException(Exception):
    """
    Represents a cyclic program exception.
    """

    def __init__(self, atom) -> None:
        """
        Creates an term malformed exception.

        :param atom: the atom
        :type atom: Atom
        """
        super().__init__("Cyclic program, cannot create the Predicate Node for "
                         "{}".format(atom))


class LiteralLayer(keras.layers.Layer):
    """
    A Layer to combine the inputs of a literal. The inputs of a literal are
    the facts of the literal and the result of rules with the literal in
    their heads.
    """

    def __init__(self, name, input_layers, literal_combining_function,
                 **kwargs):
        """
        Creates a LiteralLayer.

        :param name: the name of the layer
        :type name: str
        :param input_layers: the input layers.
        :type input_layers: List[FactLayer or RuleLayer]
        :param literal_combining_function: the literal combining function
        :type literal_combining_function: function
        :param kwargs: additional arguments
        :type kwargs: dict
        """
        # noinspection PyTypeChecker
        kwargs["name"] = name
        super(LiteralLayer, self).__init__(**kwargs)
        self.input_layers = input_layers
        self.literal_combining_function = literal_combining_function

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        if len(self.input_layers) == 1:
            return self.input_layers[0](inputs)

        results = []
        for input_layer in self.input_layers:
            results.append(input_layer(inputs))
        return self.literal_combining_function(results)

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


class FactLayer(keras.layers.Layer):
    """
    A Layer to represent the logic facts from a predicate.
    """

    def __init__(self, name, kernel, fact_combining_function, **kwargs):
        """
        Creates a PredicateLayer.

        :param name: the name of the layer
        :type name: str
        :param kernel: the data of the layer.
        :type kernel: tf.Tensor
        :param fact_combining_function: the fact combining function
        :type fact_combining_function: function
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        # noinspection PyTypeChecker
        kwargs["name"] = name
        super(FactLayer, self).__init__(**kwargs)
        self.kernel = kernel
        self.fact_combining_function = fact_combining_function

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        return self.fact_combining_function(inputs, self.kernel)

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


def compute_path_tensor(inputs, path):
    """
    Computes the `path` for the `inputs`.
    :param inputs: the inputs
    :param path: the path
    :type path: collections.Iterable[LiteralLayer]
    :return: the computed path
    :rtype: tf.Tensor
    """
    tensor = inputs
    for literal_layer in path:
        tensor = literal_layer(tensor)
    return tensor


class RuleLayer(keras.layers.Layer):
    """
    A Layer to represent a logic rule.
    """

    def __init__(self, name, paths, grounds, path_combining_function, **kwargs):
        """
        Creates a RuleLayer.

        :param name: the name of the layer
        :type name: str
        :param paths: the paths of the layer
        :type paths: List[collections.Iterable[LiteralLayer]]
        :param grounds: the grounded literals
        :type grounds: List[tf.Tensor]
        :param path_combining_function: the path combining function
        :type path_combining_function: function
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        # noinspection PyTypeChecker
        kwargs["name"] = name
        super(RuleLayer, self).__init__(**kwargs)
        self.paths = paths
        self.grounds = grounds
        self.path_combining_function = path_combining_function

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        path_result = compute_path_tensor(inputs, self.paths[0])
        for i in range(1, len(self.paths)):
            tensor = compute_path_tensor(inputs, self.paths[i])
            path_result = self.path_combining_function(path_result, tensor)
        for grounded in self.grounds:
            path_result = self.path_combining_function(path_result, grounded)
        return path_result

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


class SpecificRuleLayer(keras.layers.Layer):
    """
    A Layer to represent a rule with constants applied to it.

    It is used to extract a more specific literal from a rule inference,
    for instance the literal l(X, a), from a rule with head l(X, Y).
    """

    def __init__(self, name, rule_layer, input_constant, output_constant=None,
                 **kwargs):
        """
        Creates a SpecificRuleLayer.

        :param name: the name of the layer
        :type name: str
        :param rule_layer: the more general rule layer
        :type rule_layer: RuleLayer
        :param input_constant: the input constant
        :type input_constant: tf.Tensor
        :param output_constant: the output constant, if any
        :type output_constant: tf.Tensor
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        # noinspection PyTypeChecker
        kwargs["name"] = name
        super(SpecificRuleLayer, self).__init__(**kwargs)
        self.rule_layer = rule_layer
        self.input_constant = input_constant
        self.output_constant = output_constant

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        result = self.rule_layer(self.input_constant)
        if self.output_constant is not None:
            result = tf.matmul(result, self.output_constant)
            result = tf.reshape(result, [-1])
        return result

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(SpecificRuleLayer, self).get_config()

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

    def reverse(self):
        """
        Gets a reverse path.

        :return: the reverse path
        :rtype: RulePath
        """
        not_inverted = self.path[0].arity() != 1 and not self.inverted[0]
        path = RulePath(reversed(self.path), not_inverted)
        return path

    def __getitem__(self, item):
        return self.path.__getitem__(item)

    def __len__(self):
        return self.path.__len__()

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

    __repr__ = __str__


class NeuralLogNetwork(keras.Model):
    """
    The NeuralLog Network.
    """

    _literal_layers: Dict[Tuple[Literal, bool], LiteralLayer] = dict()
    "The literal layer by literal"

    _fact_layers: Dict[Tuple[Literal, bool], FactLayer] = dict()
    "The fact layer by literal"

    _rule_layers: Dict[Tuple[HornClause, bool], RuleLayer] = dict()
    "The rule layer by clause"

    # TODO: Create a way to define the functions in the program
    def __init__(self, program,
                 literal_combining_function=tf.math.add_n,
                 path_combining_function=tf.math.multiply,
                 edge_combining_function=tf.math.multiply,
                 edge_combining_function_2d=tf.matmul,
                 invert_fact_function=tf.transpose,
                 attribute_combine_function=tf.math.multiply):
        """
        Creates a NeuralLogNetwork.

        :param program: the neural language
        :type program: NeuralLogProgram
        :param literal_combining_function: function to combine the different
        proves of a literal (FactLayers and RuleLayers). The default is to
        sum all the proves, element-wise, by applying the `tf.math.add_n`
        function
        :type literal_combining_function: function
        :param path_combining_function: function to combine different path
        from a RuleLayer. The default is to multiply all the paths,
        element-wise, by applying the `tf.math.multiply` function
        :type path_combining_function: function
        :param edge_combining_function: function to extract the value of the
        fact based on the input. The default is the element-wise
        multiplication implemented by the `tf.math.multiply` function
        :type edge_combining_function: function
        :param edge_combining_function_2d: function to extract the value of the
        fact based on the input, for 2d facts. The default is the dot
        multiplication implemented by the `tf.matmul` function
        :type edge_combining_function_2d: function
        :param invert_fact_function: function to extract inverse of a facts.
        The default is the transpose function implemented by `tf.transpose`
        :type invert_fact_function: function
        :param attribute_combine_function: function to combine the weights
        and values of the attribute facts. The default function is the
        `tf.matmul`
        :type attribute_combine_function: function
        """
        super(NeuralLogNetwork, self).__init__(name="NeuralLogNetwork")
        self.program = program
        self.constant_size = len(self.program.iterable_constants)
        self.literal_combining_function = literal_combining_function
        self.path_combining_function = path_combining_function
        self.edge_combining_function = edge_combining_function
        self.edge_combining_function_2d = edge_combining_function_2d
        self.invert_fact_function = invert_fact_function
        self.attribute_combine_function = attribute_combine_function
        self.tensor_factory = TensorFactory(self.program)
        self.predicates = []

    # noinspection PyMissingOrEmptyDocstring
    def build(self, input_shape):
        for predicate in self.program.examples:
            predicate = self._build_literal(
                Literal(
                    Atom(predicate, *list(map(lambda x: "X{}".format(x),
                                              range(predicate.arity))))))
            self.predicates.append(predicate)

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, training=None, mask=None):
        results = []
        for predicate in self.predicates:
            results.append(predicate(inputs))
        return tuple(results)

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape(shape)

    # noinspection PyMissingOrEmptyDocstring
    def get_matrix_representation(self, atom):
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
        :rtype: tf.Tensor
        """
        return self.tensor_factory.build_atom(atom)

    # noinspection PyMissingOrEmptyDocstring
    def _build_literal(self, atom, previous_atoms=None, inverted=False):
        """
        Builds the layer for the literal.

        :param atom: the atom
        :type atom: Atom
        :param previous_atoms: the previous literals
        :type previous_atoms: Set[Atom] or None
        :param inverted: if `True`, creates the inverted literal; this is,
            a literal in the format (output, input). If `False`, creates the
            standard (input, output) literal format.
        :type inverted: bool
        :return: the predicate layer
        :rtype: LiteralLayer
        """
        # TODO: check if the literal is a function symbol
        renamed_literal = get_renamed_literal(atom)
        key = (renamed_literal, inverted)
        literal_layer = self._literal_layers.get(key, None)
        if literal_layer is None:
            if is_cyclic(renamed_literal, previous_atoms):
                raise CyclicProgramException(atom)
            inputs = [self._build_fact(renamed_literal, inverted=inverted)]
            for clause in self.program.clauses_by_predicate.get(
                    renamed_literal.predicate, []):
                substitution = get_substitution(clause.head, renamed_literal)
                if substitution is None:
                    continue
                rule = self._build_rule(clause, previous_atoms, inverted)
                if rule is None:
                    continue
                if atom.get_number_of_variables() < atom.arity():
                    layer_name = "rule_layer_{}".format(
                        get_standardised_name(clause.__str__()))
                    source = renamed_literal.terms[0 if inverted else -1]
                    destination = renamed_literal.terms[-1 if inverted else 0]
                    input_tensor = self.tensor_factory.get_one_hot_tensor(
                        source)
                    output_tensor = None
                    if destination.is_constant() and atom.arity() == 2:
                        output_tensor = self.tensor_factory.get_one_hot_tensor(
                            destination)
                        output_tensor = tf.transpose(output_tensor)
                    rule = SpecificRuleLayer(
                        layer_name, rule, input_tensor, output_tensor)
                inputs.append(rule)

            literal_layer = LiteralLayer(
                "literal_layer_{}".format(
                    get_standardised_name(renamed_literal.__str__())),
                inputs, self.literal_combining_function)
            self._literal_layers[key] = literal_layer
        return literal_layer

    def _build_rule(self, clause, previous_atoms=None, inverted=False):
        """
        Builds the Rule Node.

        :param clause: the clause
        :type clause: HornClause
        :param previous_atoms: the previous atoms
        :type previous_atoms: Set[Atom] or None
        :param inverted: if `True`, creates the layer for the inverted rule;
            this is, the rule in the format (output, input). If `False`,
            creates the layer for standard (input, output) rule format.
        :type inverted: bool
        :return: the rule layer
        :rtype: RuleLayer
        """
        key = (clause, inverted)
        rule_layer = self._rule_layers.get(key, None)
        if rule_layer is None:
            current_atoms = \
                set() if previous_atoms is None else set(previous_atoms)
            current_atoms.add(clause.head)
            paths, grounds = find_clause_paths(clause, inverted=inverted)

            layer_paths = []
            for path in paths:
                layer_path = []
                for i in range(len(path)):
                    literal_layer = self._build_literal(
                        path[i], previous_atoms, path.inverted[i])
                    layer_path.append(literal_layer)
                layer_paths.append(layer_path)

            grounded_literals = []
            for grounded in grounds:
                literal_layer = self._build_literal(grounded, previous_atoms)
                grounded_literals.append(literal_layer(None))
            layer_name = "rule_layer_{}".format(
                get_standardised_name(clause.__str__()))
            rule_layer = RuleLayer(layer_name, layer_paths, grounded_literals,
                                   self.path_combining_function)
            self._rule_layers[key] = rule_layer

        return rule_layer

    def _build_fact(self, atom, inverted=False):
        """
        Builds the fact layer for the atom.

        :param atom: the atom
        :type atom: Atom
        :param inverted: if `True`, creates the inverted fact; this is,
        a fact in the format (output, input). If `False`, creates the
        standard (input, output) fact format.
        :type inverted: bool
        :return: the fact layer
        :rtype: FactLayer
        """
        renamed_atom = get_renamed_literal(atom)
        key = (renamed_atom, inverted)
        fact_layer = self._fact_layers.get(key, None)
        if fact_layer is None:
            layer_name = "fact_layer_{}".format(
                get_standardised_name(renamed_atom.__str__()))
            tensor = self.get_matrix_representation(renamed_atom)
            edge_function = self.edge_combining_function
            if isinstance(tensor, tuple):
                tensor = self.attribute_combine_function(tensor[0], tensor[1])
            elif atom.get_number_of_variables() == 2:
                edge_function = self.edge_combining_function_2d
                if inverted:
                    tensor = self.invert_fact_function(tensor)

            shape = tensor.shape.as_list()
            if len(shape) == 2 and shape[0] > shape[-1]:
                tensor = tf.transpose(tensor)
            fact_layer = FactLayer(layer_name, tensor, edge_function)
            self._fact_layers[key] = fact_layer

        return fact_layer

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
