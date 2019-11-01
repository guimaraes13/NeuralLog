"""
Compiles the language into a neural network.
"""
import logging
from collections import deque, OrderedDict
from typing import Dict, Set, List, Tuple

import tensorflow as tf
from tensorflow.python import keras

from src.knowledge.program import NeuralLogProgram, NO_EXAMPLE_SET
from src.language.language import Atom, Term, HornClause, Literal, \
    get_renamed_literal, get_substitution, TooManyArgumentsFunction, \
    get_variable_indices, Predicate, get_variable_atom
from src.network.network_functions import get_literal_function, \
    get_combining_function
from src.network.tensor_factory import TensorFactory, \
    get_standardised_name

# Network part
# TODO: create a function to transform the examples from logic to numeric
# TODO: test everything
# QUESTION: Should we move the functional symbols to the end of the path?
#  if we do, the rule will have the same behaviour, independent of the
#  order of the literals. If we do not, we will be able to choose the intended
#  behaviour, based on the order of the literals.

# IMPROVE: test if the values of trainable iterable constants and trainable
#  variables are pointing to the same variable.
#  Skip this to the network test.

# WARNING: Do not support literals with same variable in the head of rules.
# WARNING: Do not support literals with constant numbers in the rules.
SPARSE_FUNCTION_SUFFIX = ":sparse"
ANY_PREDICATE_NAME = "any"

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
            any_literal = Literal(Atom(ANY_PREDICATE_NAME,
                                       destination, path.path_end()))
        else:
            any_literal = Literal(Atom(ANY_PREDICATE_NAME,
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


class LiteralLayer(NeuralLogLayer):
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
        super(LiteralLayer, self).__init__(name, **kwargs)
        self.input_layers = input_layers
        self.literal_combining_function = literal_combining_function

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        if len(self.input_layers) == 1:
            return self.input_layers[0](inputs)

        # results = []
        # for input_layer in self.input_layers:
        #     results.append(input_layer(inputs))
        # return self.literal_combining_function(results)
        result = self.input_layers[0](inputs)
        for input_layer in self.input_layers[1:]:
            layer_result = input_layer(inputs)
            result = self.literal_combining_function(result, layer_result)
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
        super(FunctionLayer, self).__init__(name, **kwargs)
        self.function = function
        self.inputs = inputs

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        if self.inputs is None:
            return self.function(inputs)
        return self.function(self.inputs)

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

    def __init__(self, name, aggregation_function, multiples, **kwargs):
        """
        Creates an AnyLiteralLayer

        :param name: the name of the layer
        :type name: str
        :param multiples: the tile multiples
        :type multiples: tf.Tensor
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        super(AnyLiteralLayer, self).__init__(name, **kwargs)
        self.aggregation_function = aggregation_function
        self.multiples = multiples

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        result = self.aggregation_function(inputs)
        result = tf.reshape(result, [-1, 1])
        # print("Result:", result)
        # return tf.tile(result, self.multiples)
        return result

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


class FactLayer(NeuralLogLayer):
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
        super(FactLayer, self).__init__(name, **kwargs)
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


class SpecificFactLayer(NeuralLogLayer):
    """
    A layer to represent a fact with constants applied to it.
    """

    def __init__(self, name, kernel, fact_combining_function,
                 input_constant=None,
                 input_combining_function=None, output_constant=None,
                 output_extract_function=None, **kwargs):
        """
        Creates a PredicateLayer.

        :param name: the name of the layer
        :type name: str
        :param kernel: the data of the layer.
        :type kernel: tf.Tensor
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
        self.kernel = kernel
        self.fact_combining_function = fact_combining_function
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
        result = self.fact_combining_function(input_constant, self.kernel)
        if self.output_constant is not None:
            result = self.output_extract_function(result, self.output_constant)
        return result

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    # noinspection PyTypeChecker,PyMissingOrEmptyDocstring
    def get_config(self):
        return super(SpecificFactLayer, self).get_config()

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
        :type paths: List[collections.Iterable[LiteralLayer]]
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
        super(RuleLayer, self).__init__(name, **kwargs)
        self.paths = paths
        self.grounded_layers = grounded_layers
        self.path_combining_function = path_combining_function
        self.neural_element = neutral_element

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        path_result = self._compute_path_tensor(inputs, 0)
        for i in range(1, len(self.paths)):
            tensor = self._compute_path_tensor(inputs, i)
            path_result = self.path_combining_function(path_result, tensor)
        for grounded_layer in self.grounded_layers:
            grounded_result = grounded_layer(self.neural_element)
            path_result = self.path_combining_function(path_result,
                                                       grounded_result)
        return path_result

    def _compute_path_tensor(self, inputs, index):
        """
        Computes the path for the `inputs`.

        :param inputs: the inputs
        :type inputs: tf.Tensor
        :param index: the index of the path
        :type index: int
        :return: the computed path
        :rtype: tf.Tensor
        """
        tensor = inputs
        for literal_layer in self.paths[index]:
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


class SpecificRuleLayer(NeuralLogLayer):
    """
    A layer to represent a rule with constants applied to it.

    It is used to extract a more specific literal from a rule inference,
    for instance the literal l(X, a), from a rule with head l(X, Y).
    """

    def __init__(self, name, rule_layer, input_constant,
                 inputs_combining_function, output_constant=None,
                 **kwargs):
        """
        Creates a SpecificRuleLayer.

        :param name: the name of the layer
        :type name: str
        :param rule_layer: the more general rule layer
        :type rule_layer: RuleLayer
        :param input_constant: the input constant
        :type input_constant: tf.Tensor
        :type inputs_combining_function: function
        :param inputs_combining_function: the function to combine the fixed
        input with the input of the layer
        :param output_constant: the output constant, if any
        :type output_constant: tf.Tensor
        :param kwargs: additional arguments
        :type kwargs: dict[str, Any]
        """
        super(SpecificRuleLayer, self).__init__(name, **kwargs)
        self.rule_layer = rule_layer
        self.input_constant = input_constant
        self.inputs_combining_function = inputs_combining_function
        self.output_constant = output_constant

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, **kwargs):
        input_constant = self.inputs_combining_function(self.input_constant,
                                                        inputs)
        result = self.rule_layer(input_constant)
        if self.output_constant is not None:
            result = tf.nn.embedding_lookup(result, self.output_constant)
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

    program: NeuralLogProgram
    "The NeuralLog program"

    def __init__(self, program):
        """
        Creates a NeuralLogNetwork.

        :param program: the neural language
        :type program: NeuralLogProgram
        """
        super(NeuralLogNetwork, self).__init__(name="NeuralLogNetwork")
        self.program = program
        self.constant_size = len(self.program.iterable_constants)
        self.tensor_factory = TensorFactory(self.program)
        self.predicates = OrderedDict()

        self.path_combining_function = self._get_path_combining_function()
        self.edge_combining_function = self._get_edge_combining_function()
        self.edge_combining_function_2d = None
        self.edge_combining_function_2d_sparse = None
        self._get_edge_combining_function_2d()
        self.neutral_element = self._get_edge_neutral_element()
        self.any_literal_layer = self._get_any_literal()

    def _get_edge_combining_function_2d(self):
        function_name = "edge_combining_function_2d"
        combining_function = self.program.get_parameter_value(function_name)
        self.edge_combining_function_2d = get_combining_function(
            combining_function)
        function_name += SPARSE_FUNCTION_SUFFIX
        combining_function = self.program.get_parameter_value(function_name)
        self.edge_combining_function_2d_sparse = get_combining_function(
            combining_function)

    def get_literal_negation_function(self, atom, sparse=False):
        """
        Gets the literal negation function for the atom. This function is the
        function to be applied when the atom is negated.

        The default function 1 - `a`, where `a` is the tensor representation
        of the atom.

        :param atom: the atom
        :type atom: Atom
        :param sparse: if the tensor representation of `atom` is sparse
        :type sparse: bool
        :return: the negation function
        :rtype: function
        """
        function_name = "literal_negation_function"
        if sparse:
            function_name += SPARSE_FUNCTION_SUFFIX
        name = self.program.get_parameter_value(function_name, atom.predicate)
        return get_literal_function(name)

    def get_edge_combining_function_2d(self, sparse=False):
        """
        Gets the edge combining function 2d. This is the function to extract the
        value of the fact based on the input, for 2d facts.

        The default is the dot multiplication implemented by the `tf.matmul`
        function.

        :param sparse: if the kernel tensor is sparse
        :param sparse: bool
        :return: the combining function
        :rtype: function
        """
        if sparse:
            return self.edge_combining_function_2d_sparse

        return self.edge_combining_function_2d

    def _get_path_combining_function(self):
        """
        Gets the path combining function. This is the function to combine
        different path from a RuleLayer.

        The default is to multiply all the paths, element-wise, by applying the
        `tf.math.multiply` function.

        :return: the combining function
        :rtype: function
        """
        combining_function = self.program.get_parameter_value(
            "path_combining_function")
        return get_combining_function(combining_function)

    def _get_edge_neutral_element(self):
        """
        Gets the neutral element of the edge combining function. This element is
        used to extract the tensor value of grounded literal in a rule.

        The default edge combining function is the element-wise
        multiplication. Thus, the neutral element is `1.0`, represented by
        `tf.constant(1.0)`.

        :return: the combining function
        :rtype: tf.Tensor
        """
        combining_function = self.program.get_parameter_value(
            "edge_neutral_element")
        return get_combining_function(combining_function)

    def _get_edge_combining_function(self):
        """
        Gets the edge combining function. This is the function to extract the
        value of the fact based on the input.

        The default is the element-wise multiplication implemented by the
        `tf.math.multiply` function.

        :return: the combining function
        :rtype: function
        """
        combining_function = self.program.get_parameter_value(
            "edge_combining_function")
        return get_combining_function(combining_function)

    def _get_any_literal(self):
        """
        Gets the any literal layer.

        :return: the any literal layer
        :rtype: AnyLiteralLayer
        """
        combining_function = self.program.get_parameter_value(
            "any_aggregation_function")
        function = get_combining_function(combining_function)
        return AnyLiteralLayer("literal_layer_any-X0-X1-", function,
                               tf.constant([1, self.constant_size]))

    # noinspection PyMissingOrEmptyDocstring
    def build(self, input_shape):
        for example_set in self.program.examples.values():
            for predicate in example_set:
                if predicate in self.predicates.keys():
                    continue
                predicate_layer = self._build_literal(
                    Literal(
                        Atom(predicate, *list(map(lambda x: "X{}".format(x),
                                                  range(predicate.arity))))))
                self.predicates[predicate] = predicate_layer

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, training=None, mask=None):
        results = []
        for predicate_layer in self.predicates.values():
            results.append(predicate_layer(inputs))
        # if len(results) == 1:
        #     return results[0]
        return tuple(results)

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape(shape)

    # noinspection PyMissingOrEmptyDocstring
    def get_matrix_representation(self, atom):
        """
        Builds the matrix representation for the atom.

        :param atom: the atom
        :type atom: Atom
        :raise UnsupportedMatrixRepresentation in the case the predicate is
        not convertible to matrix form
        :return: the matrix representation of the data for the given predicate
        :rtype: tf.Tensor
        """
        atom_tensor = self.tensor_factory.build_atom(atom)
        if isinstance(atom, Literal) and atom.negated:
            sparse = isinstance(atom_tensor, tf.SparseTensor)
            return self.get_literal_negation_function(atom, sparse)(atom_tensor)
        return atom_tensor

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
        :rtype: LiteralLayer or FunctionLayer
        """
        renamed_literal = get_renamed_literal(atom)
        key = (renamed_literal, inverted)
        literal_layer = self._literal_layers.get(key, None)
        if literal_layer is None:
            if atom.predicate in self.program.logic_predicates:
                literal_layer = self._build_logic_literal_layer(
                    renamed_literal, previous_atoms, inverted)
            else:
                literal_layer = self._build_function_layer(renamed_literal)
            self._literal_layers[key] = literal_layer
        return literal_layer

    def _build_logic_literal_layer(self, renamed_literal, previous_atoms,
                                   inverted):
        """
        Builds the logic literal layer.

        :param renamed_literal: the renamed literal
        :type renamed_literal: Atom
        :param previous_atoms: the previous atoms
        :type previous_atoms: list[Atom] or set[Atom]
        :param inverted: if `True`, creates the inverted literal; this is,
            a literal in the format (output, input). If `False`, creates the
            standard (input, output) literal format.
        :type inverted: bool
        :return: the literal layer
        :rtype: LiteralLayer
        """
        if is_cyclic(renamed_literal, previous_atoms):
            raise CyclicProgramException(renamed_literal)
        inputs = [self._build_fact(renamed_literal, inverted=inverted)]
        for clause in self.program.clauses_by_predicate.get(
                renamed_literal.predicate, []):
            substitution = get_substitution(clause.head, renamed_literal)
            if substitution is None:
                continue
            rule = self._build_rule(clause, previous_atoms, inverted)
            if rule is None:
                continue
            arity = renamed_literal.arity()
            if renamed_literal.get_number_of_variables() < arity:
                layer_name = "rule_layer_{}".format(
                    get_standardised_name(clause.__str__()))
                source = renamed_literal.terms[-1 if inverted else 0]
                destination = renamed_literal.terms[0 if inverted else -1]
                input_tensor = self.tensor_factory.get_one_hot_tensor(source)
                lookup = None
                if destination.is_constant() and arity == 2:
                    lookup = self.tensor_factory.get_constant_lookup(
                        destination)
                rule = SpecificRuleLayer(
                    layer_name, rule, input_tensor,
                    self._get_path_combining_function(), lookup)
            inputs.append(rule)
        combining_function = self.get_literal_combining_function(
            renamed_literal)
        return LiteralLayer(
            "literal_layer_{}".format(
                get_standardised_name(renamed_literal.__str__())), inputs,
            combining_function)

    def get_literal_combining_function(self, literal):
        """
        Gets the combining function for the `literal`. This is the function to
        combine the different proves of a literal (FactLayers and RuleLayers).

        The default is to sum all the proves, element-wise, by applying the
        `tf.math.add_n` function.

        :param literal: the literal
        :type literal: Atom
        :return: the combining function
        :rtype: function
        """
        literal_combining_function = self.program.get_parameter_value(
            "literal_combining_function", literal.predicate)
        return get_combining_function(literal_combining_function)

    def _build_function_layer(self, renamed_literal):
        """
        Builds the logic literal layer.

        :param renamed_literal: the renamed literal
        :type renamed_literal: Atom
        :return: the function layer
        :rtype: FunctionLayer
        """
        function_identifier = self.program.get_parameter_value(
            "function_value", renamed_literal.predicate)
        if function_identifier is None:
            function_identifier = renamed_literal.predicate.name
        function_value = get_literal_function(function_identifier)
        if renamed_literal.arity() > 1:
            raise TooManyArgumentsFunction(renamed_literal.predicate)
        inputs = None
        term = renamed_literal.terms[0]
        if term.is_constant():
            inputs = self.tensor_factory.get_one_hot_tensor(term)
        name = "literal_layer_{}".format(
            get_standardised_name(renamed_literal.__str__()))
        return FunctionLayer(name, function_value, inputs=inputs)

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
                    if path[i].predicate.name == ANY_PREDICATE_NAME:
                        literal_layer = self.any_literal_layer
                    else:
                        literal_layer = self._build_literal(
                            path[i], current_atoms, path.inverted[i])
                    layer_path.append(literal_layer)
                layer_paths.append(layer_path)

            grounded_layers = []
            for grounded in grounds:
                literal_layer = self._build_literal(grounded, current_atoms)
                grounded_layers.append(literal_layer)
            layer_name = "rule_layer_{}".format(
                get_standardised_name(clause.__str__()))
            rule_layer = RuleLayer(
                layer_name, layer_paths, grounded_layers,
                self.path_combining_function, self.neutral_element)
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
        renamed_literal = get_renamed_literal(atom)
        key = (renamed_literal, inverted)
        fact_layer = self._fact_layers.get(key, None)
        if fact_layer is None:
            variable_literal = get_variable_atom(atom)
            tensor = self.get_matrix_representation(variable_literal)
            edge_function = self.edge_combining_function
            is_sparse = isinstance(tensor, tf.SparseTensor)
            arity = renamed_literal.arity()
            if arity == 2:
                edge_function = self.get_edge_combining_function_2d(is_sparse)
                if inverted:
                    tensor = self.get_invert_fact_function(
                        renamed_literal)(tensor)
            elif is_sparse:
                tensor = tf.sparse.to_dense(tensor)
            layer_name = "fact_layer_{}".format(
                get_standardised_name(renamed_literal.__str__()))
            if renamed_literal.get_number_of_variables() < arity:
                input_constant = None
                output_constant = None
                source = renamed_literal.terms[-1 if inverted else 0]
                if source.is_constant() and arity == 2:
                    input_constant = self.tensor_factory.get_one_hot_tensor(
                        source)
                destination = renamed_literal.terms[0 if inverted else -1]
                output_extract_function = tf.nn.embedding_lookup
                if destination.is_constant():
                    if arity == 1:
                        output_constant = \
                            self.tensor_factory.get_constant_lookup(destination)
                    else:
                        output_extract_function = \
                            self.get_edge_combining_function_2d()
                        output_constant = \
                            self.tensor_factory.get_one_hot_tensor(destination)
                        output_constant = tf.transpose(output_constant)
                input_combining_function = self._get_path_combining_function()
                fact_layer = SpecificFactLayer(
                    layer_name, tensor, edge_function,
                    input_constant=input_constant,
                    input_combining_function=input_combining_function,
                    output_constant=output_constant,
                    output_extract_function=output_extract_function)
            else:
                fact_layer = FactLayer(layer_name, tensor, edge_function)
            self._fact_layers[key] = fact_layer

        return fact_layer

    def get_invert_fact_function(self, literal):
        """
        Gets the fact inversion function. This is the function to extract
        the inverse of a facts.

        The default is the transpose function implemented by `tf.transpose`.

        :param literal: the literal
        :type literal: Atom
        :return: the combining function
        :rtype: function
        """
        combining_function = self.program.get_parameter_value(
            "invert_fact_function", literal.predicate)
        return get_combining_function(combining_function)

    # noinspection PyTypeChecker
    def update_program(self):
        """
        Updates the program based on the learned parameters.
        """
        for atom, tensor in self.tensor_factory.variable_cache.items():
            variable_indices = get_variable_indices(atom)
            rank = len(variable_indices)
            values = tensor.numpy()
            if rank == 0:
                fact = Atom(atom.predicate, *atom.terms, weight=values)
                self.program.add_fact(fact)
            elif rank == 1:
                for i in range(self.constant_size):
                    fact = Atom(atom.predicate, *atom.terms, weight=values[i])
                    fact.terms[variable_indices[0]] = \
                        self.program.iterable_constants[i]
                    self.program.add_fact(fact)
            elif rank == 2:
                for i in range(self.constant_size):
                    for j in range(self.constant_size):
                        fact = Atom(atom.predicate, *atom.terms,
                                    weight=values[i, j])
                        fact.terms[variable_indices[0]] = \
                            self.program.iterable_constants[i]
                        fact.terms[variable_indices[1]] = \
                            self.program.iterable_constants[j]
                        self.program.add_fact(fact)


class NeuralLogDataset:
    """
    Represents a NeuralLog dataset to train a NeuralLog network.
    """

    network: NeuralLogNetwork
    "The NeuralLog program"

    examples: Dict[Term, Dict[Predicate, Dict[Term, float] or float]]

    def __init__(self, network):
        """
        Creates a NeuralLogNetwork.

        :param network: the NeuralLog network
        :type network: NeuralLogNetwork
        """
        self.network = network

    def build(self, example_set=NO_EXAMPLE_SET, sparse_features=False):
        """
        Builds the features and label to train the neural network based on
        the `example_set`.

        The labels are always a sparse tensor.

        :param example_set: the name of the set of examples
        :type example_set: str
        :param sparse_features: If `True`, the features are generate as a
        sparse tensor. If `False`, the features are generated as a dense
        tensor of indices, for each index a one hot vector creation is
        necessary.
        :type sparse_features: bool
        :return: the features and labels
        :rtype: (tf.Tensor or tf.SparseTensor, tf.SparseTensor)
        """
        # TODO: preprocess the 1D tensor to create a one hot vector of
        #  `constant_size` wide and 1.0 at the position of the 1D tensor value.
        constant_size = self.network.constant_size
        index_by_term = OrderedDict()  # type: OrderedDict[Term, int]
        predicates = []
        labels_values = []
        labels_indices = []
        index = 0
        examples = self.network.program.examples.get(example_set, OrderedDict())
        for predicate in self.network.predicates.keys():
            # for facts in examples.get(predicate, dict()).values():
            predicates.append(predicate)
            # facts = facts.values()
            facts = examples.get(predicate, dict()).values()
            values = []
            indices = []
            for fact in facts:
                weight = fact.weight
                if weight == 0.0:
                    continue
                input_term = fact.terms[0]
                term_index = index_by_term.get(input_term, None)
                if term_index is None:
                    term_index = index
                    index_by_term[input_term] = term_index
                    index += 1
                values.append(weight)
                if predicate.arity == 1:
                    indices.append([term_index])
                else:
                    output_term = fact.terms[-1]
                    indices.append(
                        [term_index,
                         self.network.program.index_for_constant(
                             output_term)])
            labels_indices.append(indices)
            labels_values.append(values)

        labels = []
        for i in range(len(predicates)):
            if predicates[i].arity == 1:
                dense_shape = [constant_size]
                empty_index = [[0]]
            else:
                dense_shape = [len(index_by_term), constant_size]
                empty_index = [[0, 0]]
            if len(labels_values[i]) == 0:
                sparse_tensor = tf.SparseTensor(indices=empty_index,
                                                values=[0.0],
                                                dense_shape=dense_shape)
            else:
                sparse_tensor = tf.SparseTensor(indices=labels_indices[i],
                                                values=labels_values[i],
                                                dense_shape=dense_shape)
            sparse_tensor = tf.sparse.reorder(sparse_tensor)
            labels.append(sparse_tensor)

        if sparse_features:
            feature_indices = []
            index = 0
            for x in index_by_term.keys():
                feature_indices.append(
                    [index, self.network.program.index_for_constant(x)])
                index += 1

            number_of_examples = len(feature_indices)
            features_shape = [number_of_examples, constant_size]
            feature_values = [1.0] * number_of_examples
            features = tf.SparseTensor(
                indices=feature_indices, values=feature_values,
                dense_shape=features_shape)
        else:
            features = tf.constant(
                list(map(
                    lambda key: self.network.program.index_for_constant(key),
                    index_by_term.keys())))

        return features, tuple(labels)
