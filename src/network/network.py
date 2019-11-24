"""
Compiles the language into a neural network.
"""
import logging
import sys
from collections import OrderedDict
from typing import Dict, Set, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.training.tracking import data_structures

from src.knowledge.program import NeuralLogProgram, NO_EXAMPLE_SET, \
    ANY_PREDICATE_NAME, find_clause_paths
from src.language.language import Atom, Term, HornClause, Literal, \
    get_renamed_literal, get_substitution, TooManyArgumentsFunction, \
    get_variable_indices, Predicate, get_renamed_atom, AtomClause
from src.network.layer_factory import LayerFactory, \
    get_standardised_name
from src.network.network_functions import get_literal_function, \
    get_combining_function, FactLayer, \
    InvertedFactLayer, SpecificFactLayer, LiteralLayer, FunctionLayer, \
    AnyLiteralLayer, RuleLayer, ExtractUnaryLiteralLayer

# Network part
# IMPROVE: to try a LALR(1) parser to improve performance
# QUESTION: Should we move the functional symbols to the end of the path?
#  if we do, the rule will have the same behaviour, independent of the
#  order of the literals. If we do not, we will be able to choose the intended
#  behaviour, based on the order of the literals.

# WARNING: Do not support literals with same variable in the head of rules.
# WARNING: Do not support literals with constant numbers in the rules.

logger = logging.getLogger()


def print_neural_log_predictions(model, neural_program, dataset,
                                 writer=sys.stdout):
    """
    Prints the predictions of `model` to `writer`.

    :param model: the model
    :type model: NeuralLogNetwork
    :param neural_program: the neural program
    :type neural_program: NeuralLogProgram
    :param dataset: the dataset
    :type dataset: tf.data.Dataset
    :param writer: the writer. Default is to print to the standard output
    :type writer: Any
    """
    for features, _ in dataset:
        y_scores = model.predict(features)
        for i in range(len(y_scores)):
            predicate, inverted = model.predicates[i]
            if inverted:
                continue
            for feature, y_score in zip(features, y_scores[i]):
                x = features.numpy()
                subject_index = np.argmax(x)
                subject = neural_program.iterable_constants[subject_index]
                if predicate.arity == 1:
                    clause = AtomClause(Atom(predicate, subject,
                                             weight=float(y_score)))
                    print(clause, file=writer)
                else:
                    clause = AtomClause(Atom(predicate, subject, "X"))
                    print("%%", clause, file=writer, sep=" ")
                    for index in range(len(y_score)):
                        object_term = neural_program.iterable_constants[index]
                        clause = AtomClause(Atom(predicate,
                                                 subject, object_term,
                                                 weight=float(y_score[index])))
                        print(clause, file=writer)
            print(file=writer)
        print(file=writer)


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
        start_column = current_clause.provenance.start_column
        clause_filename = current_clause.provenance.filename

        old_start_line = older_clause.provenance.start_line
        old_start_col = older_clause.provenance.start_column
        old_clause_filename = older_clause.provenance.filename
        logger.warning(
            "Warning: clause `%s`, defined in file: %s "
            "at %d:%d ignored. The clause has already been "
            "defined in in file: %s at %d:%d",
            current_clause, clause_filename,
            start_line, start_column,
            old_clause_filename,
            old_start_line, old_start_col
        )


class NeuralLogNetwork(keras.Model):
    """
    The NeuralLog
    Network.
    """

    _literal_layers: Dict[Tuple[Literal, bool], LiteralLayer] = dict()
    "The literal layer by literal"

    _fact_layers: Dict[Tuple[Atom, bool], FactLayer] = dict()
    "The fact layer by literal"

    _rule_layers: Dict[Tuple[HornClause, bool], RuleLayer] = dict()
    "The rule layer by clause"

    program: NeuralLogProgram
    "The NeuralLog program"

    predicates: List[Tuple[Predicate, bool]]

    def __init__(self, program, train=True):
        """
        Creates a NeuralLogNetwork.

        :param program: the neural language
        :type program: NeuralLogProgram
        :param train: if `False`, all the literals will be considered as not
        trainable/learnable, this is useful to build neural networks for
        inference only. In this way, the unknown facts will be treated as
        zeros, instead of being randomly initialized
        :type train: bool
        """
        super(NeuralLogNetwork, self).__init__(name="NeuralLogNetwork")
        self.program = program
        self.constant_size = len(self.program.iterable_constants)
        self.layer_factory = LayerFactory(self.program, train=train)
        # noinspection PyTypeChecker
        self.predicates = data_structures.NoDependency(list())
        self.predicate_layers = list()
        self.neutral_element = self._get_edge_neutral_element()

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

    # noinspection PyMissingOrEmptyDocstring
    # def build(self, input_shape):
    #     self.build_layers()
    #     super(NeuralLogNetwork, self).build(input_shape)

    def build_layers(self):
        for example_set in self.program.examples.values():
            for predicate in example_set:
                literal = Literal(Atom(predicate,
                                       *list(map(lambda x: "X{}".format(x),
                                                 range(predicate.arity)))))
                if (predicate, False) not in self.predicates:
                    predicate_layer = self._build_literal(literal)
                    if predicate.arity == 1:
                        combining_func = \
                            self.get_unary_literal_extraction_function(
                                predicate)
                        predicate_layer = ExtractUnaryLiteralLayer(
                            predicate_layer, combining_func)
                    self.predicates.append((predicate, False))
                    self.predicate_layers.append(predicate_layer)
                key = (predicate, True)
                if predicate.arity == 2 and key not in self.predicates:
                    predicate_layer = self._build_literal(literal,
                                                          inverted=True)
                    self.predicates.append(key)
                    self.predicate_layers.append(predicate_layer)

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

    # noinspection PyMissingOrEmptyDocstring
    def call(self, inputs, training=None, mask=None):
        results = []
        for predicate_layer in self.predicate_layers:
            results.append(predicate_layer(inputs))
        return tuple(results)

    # noinspection PyMissingOrEmptyDocstring
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape(shape)

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
        :rtype: LiteralLayer or src.network.network_functions.FunctionLayer
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
        inputs = [self._build_fact(renamed_literal, inverted=inverted)]
        if not is_cyclic(renamed_literal, previous_atoms):
            input_clauses = dict()  # type: dict[RuleLayer, HornClause]
            for clause in self.program.clauses_by_predicate.get(
                    renamed_literal.predicate, []):
                if is_clause_fact(clause):
                    continue
                substitution = get_substitution(clause.head, renamed_literal)
                if substitution is None:
                    continue
                rule = self._build_rule(clause, previous_atoms, inverted)
                if rule is None:
                    continue
                # TODO: Here we only use a generic rule to predict a specific
                #  fact. For instance, h(X, Y) :- ... to predict h(X, a).
                #  We should also use the other way around, use a rule
                #  h(X, a) :- ... to predict facts h(X, Y). which will return
                #  the values for h(X, a); and zero for every Y != a.
                rule = self._build_specific_rule(
                    renamed_literal, inverted, rule, substitution)
                if rule in input_clauses:
                    log_equivalent_clause(clause, input_clauses[rule])
                    continue
                input_clauses[rule] = clause
                inputs.append(rule)

        combining_func = self.get_literal_combining_function(renamed_literal)
        negation_function = None
        if isinstance(renamed_literal, Literal) and renamed_literal.negated:
            negation_function = self.get_literal_negation_function(
                renamed_literal.predicate)
        return LiteralLayer(
            "literal_layer_{}".format(
                get_standardised_name(renamed_literal.__str__())), inputs,
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
        for generic, specific in substitution.items():
            if not generic.is_constant() and specific.is_constant():
                substitution_terms[specific] = generic
        if len(substitution_terms) > 0:
            source = literal.terms[-1 if inverted else 0]
            destination = literal.terms[0 if inverted else -1]
            literal_string = literal.__str__()
            if inverted:
                literal_string = "inv_" + literal_string
            layer_name = get_standardised_name(
                "{}_specific_{}".format(rule.name, literal_string))
            input_constant = None
            input_combining_function = None
            output_constant = None
            output_extract_func = None

            if source.is_constant() and source in substitution_terms:
                input_constant = self.layer_factory.get_one_hot_tensor(source)
                input_combining_function = \
                    self.layer_factory.get_and_combining_function(predicate)
            if destination.is_constant() and destination in substitution_terms:
                output_constant = \
                    self.layer_factory.get_constant_lookup(destination)
                output_extract_func = \
                    self.layer_factory.get_output_extract_function(predicate)

            rule = SpecificFactLayer(
                layer_name, rule,
                input_constant=input_constant,
                input_combining_function=input_combining_function,
                output_constant=output_constant,
                output_extract_function=output_extract_func
            )
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
        function_value = get_literal_function(function_identifier)
        if renamed_literal.arity() > 1:
            raise TooManyArgumentsFunction(renamed_literal.predicate)
        inputs = None
        term = renamed_literal.terms[0]
        if term.is_constant():
            inputs = self.layer_factory.get_one_hot_tensor(term)
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
                        literal_layer = self._get_any_literal()
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
            rule_layer = \
                RuleLayer(
                    layer_name, layer_paths, grounded_layers,
                    self._get_path_combining_function(clause.head.predicate),
                    self.neutral_element, self.layer_factory.constant_size)
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
        renamed_atom = get_renamed_atom(atom)
        key = (renamed_atom, inverted)
        fact_layer = self._fact_layers.get(key, None)
        if fact_layer is None:
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

    # noinspection PyUnusedLocal
    def call(self, features, labels, *args, **kwargs):
        """
        Used to transform the features and examples from the sparse
        representation to dense in order to train the network.

        :param features: A dense index tensor of the features
        :type features: tf.Tensor
        :param labels: A tuple sparse tensor of labels
        :type labels: tuple[tf.SparseTensor]
        :param args: additional arguments
        :type args: list
        :param kwargs: additional arguments
        :type kwargs: dict
        :return: the features and label tensors
        :rtype: (tf.Tensor, tuple[tf.Tensor])
        """
        features = tf.one_hot(features, self.network.constant_size)
        labels = tuple(map(lambda x: tf.sparse.to_dense(x), labels))

        return features, labels

    __call__ = call

    def get_dataset(self, example_set=NO_EXAMPLE_SET, shuffle=False):
        """
        Gets the data set for the example set.

        :param example_set: the name of the example set
        :type example_set: str
        :return: the dataset
        :rtype: tf.data.Dataset
        """
        features, labels = self.build(example_set=example_set,
                                      sparse_features=False)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.map(self)
        if shuffle:
            dataset = dataset.shuffle(features.shape[0])
        return dataset

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
        constant_size = self.network.constant_size
        index_by_term = OrderedDict()  # type: OrderedDict[Term, int]
        predicates = []
        labels_values = []
        labels_indices = []
        index = 0
        examples = self.network.program.examples.get(example_set, OrderedDict())
        for predicate, inverted in self.network.predicates:
            # for facts in examples.get(predicate, dict()).values():
            predicates.append((predicate, inverted))
            # facts = facts.values()
            facts = examples.get(predicate, dict()).values()
            values = []
            indices = []
            for fact in facts:
                weight = fact.weight
                if weight == 0.0:
                    continue
                input_term = fact.terms[-1 if inverted else 0]
                term_index = index_by_term.get(input_term, None)
                if term_index is None:
                    term_index = index
                    index_by_term[input_term] = term_index
                    index += 1
                values.append(weight)
                if predicate.arity == 1:
                    indices.append([term_index, 0])
                else:
                    output_term = fact.terms[0 if inverted else -1]
                    indices.append(
                        [term_index,
                         self.network.program.index_for_constant(
                             output_term)])
            labels_indices.append(indices)
            labels_values.append(values)

        labels = []
        for i in range(len(predicates)):
            if predicates[i][0].arity == 1:
                dense_shape = [len(index_by_term), 1]
                empty_index = [[0, 0]]
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
