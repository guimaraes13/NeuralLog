"""
Compiles the language into a neural network.
"""
import logging
import sys
from typing import Dict, List, Tuple, Any

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.training.tracking import data_structures

from src.knowledge.graph import RulePathFinder
from src.knowledge.program import NeuralLogProgram, ANY_PREDICATE_NAME, \
    SimpleRulePathFinder
from src.language.language import Atom, Term, HornClause, Literal, \
    get_renamed_literal, get_substitution, get_variable_indices, Predicate, \
    get_renamed_atom, get_variable_atom, KnowledgeException
from src.network.dataset import NeuralLogDataset
from src.network.layer_factory import LayerFactory, \
    get_standardised_name
from src.network.network_functions import get_literal_function, \
    get_combining_function, FactLayer, \
    InvertedFactLayer, SpecificFactLayer, LiteralLayer, FunctionLayer, \
    AnyLiteralLayer, RuleLayer, ExtractUnaryLiteralLayer, DiagonalRuleLayer, \
    EmptyLayer, get_literal_layer, GraphRuleLayer, NeuralLogLoss

# WARNING: Do not support literals with same variable in the head of rules.
# WARNING: Do not support constants in the head of rules.
# WARNING: Do not support literals with constant numbers in the rules.

# WARNING: For now, we only use a generic rule to predict a specific
#  fact. For instance, h(X, Y) :- ... to predict h(X, a).
#  We should also use the other way around, use a rule
#  h(X, a) :- ... to predict facts h(X, Y). which will return
#  the values for h(X, a); and zero for every Y != a.

# WARNING: We assume that all literals with arity higher than two have
#  distinct terms and the output of those literals are always the last term.

# WARNING: We do not support facts with arity bigger than two. Since the
#  current version of tensorflow does not have operation to efficiently
#  handle sparse tensor with more than two dimensions, we would have to
#  create such facts as dense tensors, which would consume too much memory.

logger = logging.getLogger(__name__)


class LossMaskWrapper:
    """
    A mask wrapper for the loss function to mask the values of unknown labels.

    It multiplies the output of the network by the square of the labels. In
    order to this method work, the labels must be: `1`, for positive examples;
    `-1`, for negative examples; and `0`, for unknown examples.

    In this way, the square of the labels will be `1` for the positive and
    negative examples; and `0`, for the unknown examples. When multiplied by
    the prediction, the predictions of the unknown examples will be zero,
    thus, having no error and no gradient for those examples. While the
    predictions of the known examples will remain the same.
    """

    def __init__(self, loss_function, label_function=None):
        """
        Creates a loss mask wrapper.

        :param loss_function: the loss function to wrap.
        :type loss_function: function
        """
        self.loss_function = loss_function
        self.function = keras.losses.get(loss_function)
        self.label_function = label_function
        self.__name__ = self.function.__name__

    def call(self, y_true, y_pred):
        """
        The wrapped function.

        :param y_true: the true labels
        :type y_true: tf.Tensor, list[tf.Tensor], np.ndarray, list[np.ndarray]
        :param y_pred: the predictions
        :type y_pred: tf.Tensor, list[tf.Tensor], np.ndarray, list[np.ndarray]
        :return: the wrapped function
        :rtype: function
        """
        if isinstance(y_pred, list):
            new_y_pred = []
            for i in range(len(y_pred)):
                mask = tf.square(y_true[i])
                new_y_pred.append(y_pred[i] * mask)
        else:
            mask = tf.square(y_true)
            new_y_pred = y_pred * mask
        if self.label_function is not None:
            y_true = self.label_function(y_true)
        return self.function(y_true, new_y_pred)

    __call__ = call

    def __str__(self):
        return "{}({})".format(self.__class__.__name__,
                               self.loss_function.__str__())

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__,
                               self.loss_function.__repr__())


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


class CyclicProgramException(KnowledgeException):
    """
    Represents a cyclic program exception.
    """

    def __init__(self, atom):
        """
        Creates an term malformed exception.

        :param atom: the atom
        :type atom: Atom
        """
        super().__init__("Cyclic program, cannot create the Predicate Node for "
                         "{}".format(atom))


def is_clause_fact(clause):
    """
    Returns true if the clause degenerates to a fact, this happens when the
    head of the clause is equals to the the body, for instance:
    `h(X, Y) :- h(X, Y).`. This kind of clauses can be ignored since it is
    already represented by the facts of `h(X, Y)`.

    :param clause: the clause
    :type clause: HornClause
    :return: `True` if the the clause degenerates to a fact; `False` otherwise.
    :rtype: bool
    """
    if len(clause.body) != 1:
        return False
    body = clause.body[0]
    if body.negated:
        return False

    return clause.head.predicate == body.predicate and \
           clause.head.terms == body.terms


def log_equivalent_clause(current_clause, older_clause):
    """
    Logs the redundant clause.

    :param current_clause: the current clause
    :type current_clause: HornClause
    :param older_clause: the older clause
    :type older_clause: HornClause
    """
    if logger.isEnabledFor(logging.WARNING):
        if older_clause is None:
            return
        start_line = current_clause.provenance.start_line
        clause_filename = current_clause.provenance.filename

        old_start_line = older_clause.provenance.start_line
        old_clause_filename = older_clause.provenance.filename
        logger.warning(
            "Warning: clause `%s`, defined in file: %s "
            "at %d ignored. The clause has already been "
            "defined in in file: %s at %d.",
            current_clause, clause_filename,
            start_line,
            old_clause_filename,
            old_start_line
        )


def arity_bigger_than(clause, value):
    """
    Returns `True`, if the arity of the `clause` is bigger than `value`. The
    arity of a clause is defined by the maximum arity of the literals in the
    clause.

    :param clause: the clause
    :type clause: HornClause
    :param value: the value
    :type value: int
    :return: `True`, if the arity of the `clause` is bigger than `value`;
    otherwise, `False`
    :rtype: bool
    """
    if clause.head.arity() > value:
        return True

    for atom in clause.body:
        if atom.arity() > value:
            return True

    return False


class NeuralLogNetwork(keras.Model):
    """
    The NeuralLog Network.
    """

    # noinspection PyTypeChecker
    def __init__(self, dataset, train=True, inverse_relations=True,
                 regularizer=None):
        """
        Creates a NeuralLogNetwork.

        :param dataset: the NeuralLog dataset
        :type dataset: NeuralLogDataset
        :param train: if `False`, all the literals will be considered as not
        trainable/learnable, this is useful to build neural networks for
        inference only. In this way, the unknown facts will be treated as
        zeros, instead of being randomly initialized
        :param inverse_relations: if `True`, also creates the layers for the
        inverse relations.
        :type inverse_relations: bool
        :type train: bool
        :param regularizer: the regularizer
        :type regularizer: callable
        """
        super(NeuralLogNetwork, self).__init__(name="NeuralLogNetwork")

        self._literal_layers: Dict[Tuple[Literal, bool], LiteralLayer] = \
            data_structures.NoDependency(dict())
        "The literal layer by literal"

        self._fact_layers: Dict[Tuple[Atom, bool], FactLayer] = \
            data_structures.NoDependency(dict())
        "The fact layer by literal"

        self._rule_layers: Dict[Tuple[HornClause, bool], RuleLayer] = \
            data_structures.NoDependency(dict())
        "The rule layer by clause"

        self._function_by_predicate: Dict[Predicate, Any] = \
            data_structures.NoDependency(dict())
        "The function by predicate"

        self.program: NeuralLogProgram
        "The NeuralLog program"

        self.predicates: List[Tuple[Predicate, bool]]

        self.dataset = dataset
        self.program = dataset.program
        self.layer_factory = LayerFactory(
            self.program, train=train, regularizer=regularizer)
        # noinspection PyTypeChecker
        self.predicates = data_structures.NoDependency(list())
        self.predicate_layers = list()
        self.neutral_element = self._get_edge_neutral_element()
        self.neutral_element = tf.reshape(self.neutral_element, [1, 1])
        self.inverse_relations = inverse_relations
        self.empty_layer = EmptyLayer("empty")
        self.input_sizes = []
        self.loss_traced_objects = None

    # noinspection PyMissingOrEmptyDocstring
    def compile(self, *args, **kwargs):
        self._build_neural_log_loss(**kwargs)
        return super().compile(*args, **kwargs)

    def _build_neural_log_loss(self, **kwargs):
        my_loss = kwargs.get("loss", None)
        if isinstance(my_loss, LossMaskWrapper):
            my_loss = my_loss.function
        if isinstance(my_loss, NeuralLogLoss):
            tensors = my_loss.predicate_parameters()
            for key, value in tensors.items():
                atom = get_variable_atom(value)
                tensors[key] = self.layer_factory.build_atom(atom).get_kernel()
            self.loss_traced_objects = tensors
            my_loss.build(**tensors)

    def get_recursion_depth(self, predicate=None):
        """
        Gets the maximum recursion depth for the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the maximum recursion depth
        :rtype: int
        """
        value = self.program.get_parameter_value("recursion_depth", predicate)
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 15 * value))
        return value

    def get_literal_negation_function(self, predicate):
        """
        Gets the literal negation function for the atom. This function is the
        function to be applied when the atom is negated.

        The default function 1 - `a`, where `a` is the tensor representation
        of the atom.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the negation function
        :rtype: function
        """
        name = self.program.get_parameter_value("literal_negation_function",
                                                predicate)
        return get_literal_function(name)

    def _get_path_combining_function(self, predicate=None):
        """
        Gets the path combining function. This is the function to combine
        different path from a RuleLayer.

        The default is to multiply all the paths, element-wise, by applying the
        `tf.math.multiply` function.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the combining function
        :rtype: function
        """
        combining_function = self.program.get_parameter_value(
            "path_combining_function", predicate)
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

    def _get_any_literal(self):
        """
        Gets the any literal layer.

        :return: the any literal layer
        :rtype: EndAnyLiteralLayer
        """
        combining_function = self.program.get_parameter_value(
            "any_aggregation_function")
        function = get_combining_function(combining_function)
        return AnyLiteralLayer("literal_layer_any-X0-X1-", function)

    def build_layers(self):
        """
        Builds the layers of the network.
        """
        self.input_sizes = []
        for predicate, inverted in self.dataset.get_target_predicates():
            self.input_sizes.append(max(predicate.arity - 1, 1))
            logger.debug("Building output layer for predicate: %s", predicate)
            literal = Literal(Atom(
                predicate, *list(map(
                    lambda x: "X{}".format(x), range(predicate.arity)))))
            predicate_layer = self._build_literal(
                literal, dict(), inverted=inverted)
            if predicate.arity == 1:
                combining_func = self.get_unary_literal_extraction_function(
                    predicate)
                predicate_layer = ExtractUnaryLiteralLayer(
                    predicate_layer, combining_func)
            # noinspection PyUnresolvedReferences
            self.predicates.append((predicate, inverted))
            self.predicate_layers.append(predicate_layer)

        if len(self.predicate_layers) == 1:
            self.call = self.call_single_input
        else:
            # noinspection PyAttributeOutsideInit
            self.call = self.call_multiples_inputs

    def get_unary_literal_extraction_function(self, predicate):
        """
        Gets the unary literal extraction function. This is the function to
        extract the value of unary prediction.

        The default is the dot multiplication, implemented by the
        `tf.matmul`, applied to the transpose of the literal prediction.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the combining function
        :rtype: function
        """
        combining_function = self.program.get_parameter_value(
            "unary_literal_extraction_function", predicate)
        return get_combining_function(combining_function)

    # noinspection PyMissingOrEmptyDocstring,PyUnusedLocal
    def call_single_input(self, inputs, training=None, mask=None):
        return self.predicate_layers[0](inputs),

    # noinspection PyMissingOrEmptyDocstring,PyUnusedLocal
    def call_multiples_inputs(self, inputs, training=None, mask=None):
        results = []
        offset = 0
        for layer, size in zip(self.predicate_layers, self.input_sizes):
            if size > 1:
                layer_input = inputs[offset:offset + size]
            else:
                layer_input = inputs[offset]
            results.append(layer(layer_input))
            offset += size
        return tuple(results)

    # noinspection PyMissingOrEmptyDocstring
    # def compute_output_shape(self, input_shape):
    #     shape = tf.TensorShape(input_shape).as_list()
    #     return tf.TensorShape(shape)

    # noinspection PyMissingOrEmptyDocstring
    def _build_literal(self, atom, predicates_depths, inverted=False):
        """
        Builds the layer for the literal.

        :param atom: the atom
        :type atom: Atom
        :param predicates_depths: the depths of the predicates
        :type predicates_depths: dict[Predicate, int]
        :param inverted: if `True`, creates the inverted literal; this is,
            a literal in the format (output, input). If `False`, creates the
            standard (input, output) literal format.
        :type inverted: bool
        :return: the predicate layer
        :rtype: LiteralLayer or src.network.network_functions.FunctionLayer
        """
        renamed_literal = get_renamed_literal(atom)
        key = (renamed_literal, inverted)
        literal_layer = self._literal_layers.get(key, None)
        if literal_layer is None:
            if atom.predicate in self.program.logic_predicates:
                logger.debug("Building layer for literal: %s", renamed_literal)
                predicates_depths.setdefault(atom.predicate, -1)
                predicates_depths[atom.predicate] += 1
                literal_layer = self._build_logic_literal_layer(
                    renamed_literal, predicates_depths, inverted)
                predicates_depths[atom.predicate] -= 1
            else:
                logger.debug("Building layer for function: %s", renamed_literal)
                literal_layer = self._build_function_layer(renamed_literal)
            self._literal_layers[key] = literal_layer
        return literal_layer

    def _build_logic_literal_layer(self, renamed_literal, predicates_depths,
                                   inverted):
        """
        Builds the logic literal layer.

        :param renamed_literal: the renamed literal
        :type renamed_literal: Atom
        :param predicates_depths: the depths of the predicates
        :type predicates_depths: dict[Predicate, int]
        :param inverted: if `True`, creates the inverted literal; this is,
            a literal in the format (output, input). If `False`, creates the
            standard (input, output) literal format.
        :type inverted: bool
        :return: the literal layer
        :rtype: LiteralLayer
        """
        predicate = renamed_literal.predicate
        depth = predicates_depths[predicate]
        if depth < self.get_recursion_depth(predicate) + 1:
            inputs = []
            if (predicate in self.program.facts_by_predicate or
                    predicate in self.program.trainable_predicates):
                inputs = [self._build_fact(renamed_literal, inverted=inverted)]
            input_clauses = dict()  # type: Dict[RuleLayer, HornClause]
            for clause in self.program.clauses_by_predicate.get(
                    predicate, []):
                if is_clause_fact(clause):
                    continue
                substitution = get_substitution(clause.head, renamed_literal)
                if substitution is None:
                    continue
                rule = self._build_high_arity_rule(
                    clause, predicates_depths, inverted)
                if rule is None:
                    continue
                rule = self._build_specific_rule(
                    renamed_literal, inverted, rule, substitution)
                if rule in input_clauses:
                    log_equivalent_clause(clause, input_clauses[rule])
                    continue
                input_clauses[rule] = clause
                inputs.append(rule)
        else:
            inputs = [self.empty_layer]

        combining_func = self.get_literal_combining_function(renamed_literal)
        negation_function = None
        if isinstance(renamed_literal, Literal) and renamed_literal.negated:
            negation_function = self.get_literal_negation_function(
                predicate)
        return LiteralLayer(
            "literal_layer_{}_{}".format(
                get_standardised_name(renamed_literal.__str__()), depth),
            inputs,
            combining_func, negation_function=negation_function)

    def _build_specific_rule(self, literal, inverted, rule, substitution):
        """
        Builds a specific rule from a more generic one.

        :param literal: the literal
        :type literal: Atom
        :param inverted: if `True`, creates the inverted literal; this is,
            a literal in the format (output, input). If `False`, creates the
            standard (input, output) literal format.
        :type inverted: bool
        :param rule: the general rule
        :type rule: RuleLayer
        :param substitution: the dictionary with the substitution from the
        generic term to the specific one
        :type substitution: dict[Term, Term]
        :return: the specific rule
        :rtype: SpecificFactLayer
        """
        predicate = literal.predicate
        substitution_terms = dict()
        last_term = None
        equal_terms = False
        for generic, specific in substitution.items():
            equal_terms = last_term == specific or last_term is None
            last_term = specific
            if not generic.is_constant() and specific.is_constant():
                substitution_terms[specific] = generic

        if len(substitution_terms) > 0:
            output_index = 0 if inverted else -1
            sources = list(literal.terms)
            source_indices = list(range(len(literal.terms)))
            if len(sources) > 1:
                destination = sources.pop(output_index)
                destination_index = source_indices.pop(output_index)
            else:
                destination = sources[0]
                destination_index = 0
            literal_string = literal.__str__()
            if inverted:
                literal_string = "inv_" + literal_string
            layer_name = get_standardised_name(
                "{}_specific_{}".format(rule.name, literal_string))
            input_constants = []
            input_combining_functions = []
            output_constant = None
            output_extract_func = None

            for source, index in zip(sources, source_indices):
                if source.is_constant() and source in substitution_terms:
                    input_constants.append(
                        self.layer_factory.get_one_hot_tensor(literal, index))
                    input_combining_functions.append(
                        self.layer_factory.get_and_combining_function(predicate)
                    )
            if destination.is_constant() and destination in substitution_terms:
                output_constant = self.layer_factory.get_constant_lookup(
                    literal, destination_index)
                output_extract_func = \
                    self.layer_factory.get_output_extract_function(predicate)
            rule = SpecificFactLayer(
                layer_name, rule,
                input_constants=input_constants,
                input_combining_functions=input_combining_functions,
                output_constant=output_constant,
                output_extract_function=output_extract_func
            )
        elif equal_terms and predicate.arity > 1:
            rule = DiagonalRuleLayer(
                rule, self.layer_factory.get_and_combining_function(predicate))
        return rule

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
        :rtype: src.network.network_functions.FunctionLayer
        """
        function_identifier = self.program.get_parameter_value(
            "function_value", renamed_literal.predicate)
        if function_identifier is None:
            function_identifier = renamed_literal.predicate.name
        function_value = self._get_predicate_function(
            renamed_literal.predicate, function_identifier)
        inputs = None
        term = renamed_literal.terms[0]
        if term.is_constant():
            inputs = self.layer_factory.get_one_hot_tensor(renamed_literal, 0)
        name = "literal_layer_{}".format(
            get_standardised_name(renamed_literal.__str__()))
        return FunctionLayer(name, function_value, inputs=inputs)

    def _get_predicate_function(self, predicate, function_identifier):
        """
        Gets the predicate function for the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :param function_identifier: the function identifier
        :type function_identifier: str or dict
        :return: the predicate function
        :rtype: function
        """
        function_value = self._function_by_predicate.get(predicate, None)
        if function_value is None:
            try:
                function_value = get_literal_function(function_identifier)
            except (ValueError, TypeError):
                function_value = get_literal_layer(function_identifier)
            self._function_by_predicate[predicate] = function_value
        return function_value

    def _build_rule(self, clause, predicates_depths, inverted=False):
        """
        Builds the Rule Node.

        :param clause: the clause
        :type clause: HornClause
        :param predicates_depths: the depths of the predicates
        :type predicates_depths: dict[Predicate, int]
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
            logger.debug("Building layer for rule: %s", clause)
            rule_path_finder = SimpleRulePathFinder(clause)
            paths, grounds = rule_path_finder.find_clause_paths(inverted)

            layer_paths = []
            for path in paths:
                layer_path = []
                for i in range(len(path)):
                    if path[i].predicate.name == ANY_PREDICATE_NAME:
                        literal_layer = self._get_any_literal()
                    else:
                        literal_layer = self._build_literal(
                            path[i], predicates_depths, path.inverted[i])
                    layer_path.append(literal_layer)
                layer_paths.append(layer_path)

            grounded_layers = []
            for grounded in grounds:
                literal_layer = self._build_literal(grounded, predicates_depths)
                grounded_layers.append(literal_layer)
            layer_name = "rule_layer_{}".format(
                get_standardised_name(clause.__str__()))
            rule_layer = \
                RuleLayer(
                    layer_name, layer_paths, grounded_layers,
                    self._get_path_combining_function(clause.head.predicate),
                    self.neutral_element)
            self._rule_layers[key] = rule_layer

        return rule_layer

    def _build_high_arity_rule(self, clause, predicates_depths, inverted=False):
        """
        Builds the Rule Node for rules with arity bigger than two.

        :param clause: the clause
        :type clause: HornClause
        :param predicates_depths: the depths of the predicates
        :type predicates_depths: dict[Predicate, int]
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
            logger.debug("Building layer for rule: %s", clause)
            rule_path_finder = RulePathFinder(clause)
            rule_graph = rule_path_finder.find_clause_paths(-1)

            literal_layers = dict()
            for edge in rule_graph.edges:
                if edge.literal.predicate.name == ANY_PREDICATE_NAME:
                    literal_layer = self._get_any_literal()
                else:
                    literal_layer = self._build_literal(
                        edge.literal, predicates_depths, edge.is_inverted())
                literal_layers[edge] = literal_layer

            grounded_layers = []
            for grounded in rule_graph.grounds:
                literal_layer = self._build_literal(grounded, predicates_depths)
                grounded_layers.append(literal_layer)

            layer_name = "rule_layer_{}_{}".format(
                get_standardised_name(clause.__str__()),
                predicates_depths[clause.head.predicate])
            rule_layer = GraphRuleLayer(
                layer_name, clause, rule_graph, literal_layers, grounded_layers,
                self._get_path_combining_function(clause.head.predicate),
                self.layer_factory.get_and_combining_function(),
                self.neutral_element)
            self._rule_layers[key] = rule_layer

        return rule_layer

    def _create_layer_for_edge(self, edge):
        pass

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
        renamed_atom = get_renamed_atom(atom)
        key = (renamed_atom, inverted)
        fact_layer = self._fact_layers.get(key, None)
        if fact_layer is None:
            logger.debug("Building layer for fact: %s", renamed_atom)
            fact_layer = self.layer_factory.build_atom(renamed_atom)
            if inverted:
                fact_layer = InvertedFactLayer(
                    fact_layer, self.layer_factory, atom.predicate)
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
        for atom, tensor in self.layer_factory.variable_cache.items():
            variable_indices = get_variable_indices(atom)
            rank = len(variable_indices)
            values = tensor.numpy()
            size_0 = self.program.get_constant_size(atom.predicate, 0)
            if rank == 0:
                fact = Atom(atom.predicate, *atom.terms, weight=values)
                self.program.add_fact(fact)
            elif rank == 1:
                for i in range(size_0):
                    fact = Atom(atom.predicate, *atom.terms, weight=values[i])
                    fact.terms[variable_indices[0]] = \
                        self.program.get_constant_by_index(
                            atom.predicate, 0, i)
                    self.program.add_fact(fact)
            elif rank == 2:
                size_1 = self.program.get_constant_size(atom.predicate, 1)
                for i in range(size_0):
                    for j in range(size_1):
                        fact = Atom(atom.predicate, *atom.terms,
                                    weight=values[i, j])
                        fact.terms[variable_indices[0]] = \
                            self.program.get_constant_by_index(
                                atom.predicate, 0, i)
                        fact.terms[variable_indices[1]] = \
                            self.program.get_constant_by_index(
                                atom.predicate, 1, j)
                        self.program.add_fact(fact)
