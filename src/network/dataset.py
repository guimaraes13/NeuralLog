"""
Handles the examples.
"""
import collections
import logging
import sys
from abc import abstractmethod, ABC
from collections import OrderedDict, deque
from functools import partial
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from bert.tokenization.bert_tokenization import FullTokenizer

from src.knowledge.examples import Examples
from src.knowledge.program import NeuralLogProgram, NO_EXAMPLE_SET, \
    get_predicate_from_string
from src.language.language import AtomClause, Atom, Predicate, \
    get_constant_from_string, get_term_from_string, TermType, Constant, Quote
from src.network import registry

PARTIAL_WORD_PREFIX = "##"

logger = logging.getLogger(__name__)

dataset_classes = dict()


def neural_log_dataset(identifier):
    """
    A decorator for NeuralLog dataset.

    :param identifier: the identifier of the function
    :type identifier: str
    :return: the decorated function
    :rtype: function
    """
    return lambda x: registry(x, identifier, dataset_classes)


def get_dataset_class(identifier):
    """
    Returns the class of the dataset based on the `identifier`.

    :param identifier: the identifier
    :type identifier: str
    :return: the dataset class
    :rtype: function
    """
    return dataset_classes.get(identifier, DefaultDataset)


# noinspection PyTypeChecker,DuplicatedCode
def print_neural_log_predictions(model, neural_program, neural_dataset, dataset,
                                 writer=sys.stdout, dataset_name=None,
                                 print_batch_header=False):
    """
    Prints the predictions of `model` to `writer`.

    :param model: the model
    :type model: NeuralLogNetwork
    :param neural_dataset: the NeuralLog dataset
    :type neural_dataset: NeuralLogDataset
    :param neural_program: the neural program
    :type neural_program: NeuralLogProgram
    :param dataset: the dataset
    :type dataset: tf.data.Dataset
    :param writer: the writer. Default is to print to the standard output
    :type writer: Any
    :param dataset_name: the name of the dataset
    :type dataset_name: str
    :param print_batch_header: if `True`, prints a commented line before each
    batch
    :type print_batch_header: bool
    """
    count = 0
    batches = None
    empty_entry = None
    fix = 0
    if isinstance(neural_dataset, SequenceDataset):
        if print_batch_header and dataset_name is not None:
            batches = list(neural_program.mega_examples[dataset_name].keys())
        empty_entry = neural_dataset.empty_word_index
    for features, _ in dataset:
        if print_batch_header:
            batch = batches[count] if batches is not None else count
            print("%% Batch:", batch, file=writer, sep="\t")
            count += 1
        y_scores = model.predict(features)
        if len(model.predicates) == 1:
            # if not isinstance(neural_dataset, WordCharDataset):
            y_scores = [y_scores]
            if model.predicates[0][0].arity < 3:
                features = [features]
        for i in range(len(model.predicates)):
            predicate, inverted = model.predicates[i]
            if isinstance(neural_dataset, WordCharDataset):
                predicate = Predicate(predicate.name, predicate.arity - 1)
            if inverted:
                continue
            row_scores = y_scores[i]
            if len(row_scores.shape) == 3:
                row_scores = np.squeeze(row_scores, axis=1)
            for j in range(len(row_scores)):
                y_score = row_scores[j]
                x = []
                subjects = []
                stop = False
                offset = sum(model.input_sizes[:i])
                for k in range(model.input_sizes[i] + fix):
                    x_k = features[offset + k][j].numpy()
                    if x_k.dtype == np.float32:
                        if np.max(x_k) == 0:
                            stop = True
                            break
                        arg_max = np.argmax(x_k)
                        if arg_max == empty_entry:
                            stop = True
                            break
                    else:
                        arg_max = x_k[0]
                        if arg_max < 0 or arg_max == empty_entry:
                            stop = True
                            break
                    subjects.append(neural_program.get_constant_by_index(
                        predicate, k, arg_max))
                    x.append(x_k)
                offset += model.input_sizes[i]
                if stop:
                    continue

                if predicate.arity == 1:
                    clause = AtomClause(Atom(predicate, subjects[0],
                                             weight=float(y_score)))
                    print(clause, file=writer)
                else:
                    clauses = []
                    for index in range(len(y_score)):
                        object_term = neural_program.get_constant_by_index(
                            predicate, -1, index)
                        prediction = Atom(predicate, *subjects, object_term,
                                          weight=float(y_score[index]))
                        if dataset_name is not None and \
                                not neural_dataset.has_example_key(
                                    prediction.simple_key()):
                            continue
                        clauses.append(AtomClause(prediction))

                    if len(clauses) > 0:
                        clause = AtomClause(Atom(predicate, *subjects, "X"))
                        print("%%", clause, file=writer, sep=" ")
                        for clause in sorted(
                                clauses,
                                key=lambda c: c.atom.weight,
                                reverse=True):
                            print(clause, file=writer)
                        print(file=writer)
            # print(file=writer)


# noinspection DuplicatedCode
def _print_word_char_predictions(model, neural_program, neural_dataset, dataset,
                                 writer=sys.stdout, dataset_name=None,
                                 print_batch_header=False):
    """
    Prints the predictions of `model` to `writer`.

    :param model: the model
    :type model: NeuralLogNetwork
    :param neural_dataset: the NeuralLog dataset
    :type neural_dataset: WordCharDataset
    :param neural_program: the neural program
    :type neural_program: NeuralLogProgram
    :param dataset: the dataset
    :type dataset: tf.data.Dataset
    :param writer: the writer. Default is to print to the standard output
    :type writer: Any
    :param dataset_name: the name of the dataset
    :type dataset_name: str
    :param print_batch_header: if `True`, prints a commented line before each
    batch
    :type print_batch_header: bool
    """
    count = 0
    batches = None
    fix = -1
    if print_batch_header and dataset_name is not None:
        batches = list(neural_program.mega_examples[dataset_name].keys())
    empty_entry = neural_dataset.empty_word_index
    for features, _ in dataset:
        if print_batch_header:
            batch = batches[count] if batches is not None else count
            print("%% Batch:", batch, file=writer, sep="\t")
            count += 1
        y_scores = model.predict(features)
        if len(model.predicates) == 1:
            # if not isinstance(neural_dataset, WordCharDataset):
            y_scores = [y_scores]
            if model.predicates[0][0].arity < 3:
                features = [features]
        for i in range(len(model.predicates)):
            predicate, inverted = model.predicates[i]
            predicate = Predicate(predicate.name, predicate.arity - 1)
            row_scores = y_scores[i]
            if len(row_scores.shape) == 3:
                row_scores = np.squeeze(row_scores, axis=1)
            for j in range(len(row_scores)):
                y_score = row_scores[j]
                x = []
                subjects = []
                stop = False
                offset = sum(model.input_sizes[:i])
                for k in range(model.input_sizes[i] + fix):
                    x_k = features[offset + k][j].numpy()
                    if x_k.dtype == np.float32:
                        if np.max(x_k) == 0:
                            stop = True
                            break
                        arg_max = np.argmax(x_k)
                        if arg_max == empty_entry:
                            stop = True
                            break
                    else:
                        arg_max = x_k[0]
                        if arg_max < 0 or arg_max == empty_entry:
                            stop = True
                            break
                    subjects.append(neural_program.get_constant_by_index(
                        predicate, k, arg_max))
                    x.append(x_k)
                offset += model.input_sizes[i]
                if stop:
                    continue

                last_feature = features[offset - 1][j].numpy()
                subject_string = "\""
                for k in last_feature:
                    if k == neural_dataset.empty_char_index:
                        break
                    subject_string += neural_program.get_constant_by_index(
                        neural_dataset.character_predicate,
                        neural_dataset.character_predicate_index,
                        k
                    ).value
                subject_string += "\""
                subjects[-1] = get_term_from_string(subject_string)

                if predicate.arity == 1:
                    clause = AtomClause(Atom(predicate, subjects[0],
                                             weight=float(y_score)))
                    print(clause, file=writer)
                else:
                    clauses = []
                    for index in range(len(y_score)):
                        object_term = neural_program.get_constant_by_index(
                            predicate, -1, index)
                        prediction = Atom(predicate, *subjects, object_term,
                                          weight=float(y_score[index]))
                        if dataset_name is not None and \
                                not neural_dataset.has_example_key(
                                    prediction.simple_key()):
                            continue
                        clauses.append(AtomClause(prediction))

                    if len(clauses) > 0:
                        clause = AtomClause(Atom(predicate, *subjects, "X"))
                        print("%%", clause, file=writer, sep=" ")
                        for clause in sorted(
                                clauses,
                                key=lambda c: c.atom.weight,
                                reverse=True):
                            print(clause, file=writer)
                        print(file=writer)
            # print(file=writer)


def get_predicate_indices(predicate, inverted):
    """
    Gets the indices of the predicate's input and output.

    :param predicate: the predicate
    :type predicate: Predicate
    :param inverted: if the predicate is inverted
    :type inverted: bool
    :return: the input and output indices
    :rtype: (list[int], int)
    """
    if predicate.arity == 1:
        input_index = [0]
        output_index = 0
    elif predicate.arity == 2:
        if inverted:
            input_index = [1]
            output_index = 0
        else:
            input_index = [0]
            output_index = 1
    else:
        input_index = [x for x in range(predicate.arity - 1)]
        output_index = predicate.arity - 1

    return input_index, output_index


# noinspection DuplicatedCode
def viterbi(potentials, transition_matrix, initial_distribution=None):
    """
    Computes the best path given based on the Viterbi algorithm.

    :param potentials: the emission of the neural network
    :type potentials: np.ndarray
    :param transition_matrix: the transition matrix
    :type transition_matrix: np.ndarray
    :param initial_distribution: the probabilities of the first state,
    it assumes an uniform probability, if `None`.
    :type initial_distribution: np.ndarray
    :return: the best path
    :rtype: np.ndarray
    """
    sequence_length, number_of_tags = potentials.shape
    state_probabilities = np.zeros((sequence_length, number_of_tags),
                                   dtype=np.float64)
    best_paths = np.zeros((sequence_length, number_of_tags), dtype=np.int32)
    best_path = np.zeros(sequence_length, dtype=np.int32)

    if initial_distribution is None:
        # initial_distribution = np.full(number_of_tags, 1.0 / number_of_tags)
        initial_distribution = np.full(number_of_tags, 1.0)

    state_probabilities[0, :] = potentials[0, :] * initial_distribution

    transpose = transition_matrix.transpose()
    for i in range(1, sequence_length):
        prev_p = state_probabilities[i - 1, :] * transpose
        best_previous_nodes = np.argmax(prev_p, axis=1)
        state_probabilities[i] = np.max(prev_p, axis=1)
        state_probabilities[i] *= potentials[i, :]
        best_paths[i, :] = best_previous_nodes

    best_path[-1] = np.argmax(state_probabilities[-1, :])
    for i in reversed(range(1, sequence_length)):
        best_path[i - 1] = best_paths[i, best_path[i]]

    return best_path


# noinspection DuplicatedCode
def log_viterbi(potentials, transition_matrix, initial_distribution=None):
    """
    Computes the best path given based on the Viterbi algorithm.

    This version uses sum instead of multiplication and assumes that both
    `transition_matrix` and `emission` are the log of the probabilities.

    :param potentials: the emission of the neural network
    :type potentials: np.ndarray
    :param transition_matrix: the transition matrix
    :type transition_matrix: np.ndarray
    :param initial_distribution: the probabilities of the first state,
    it assumes an uniform probability, if `None`.
    :type initial_distribution: np.ndarray
    :return: the best path
    :rtype: np.ndarray
    """
    sequence_length, number_of_tags = potentials.shape
    state_probabilities = np.zeros((sequence_length, number_of_tags),
                                   dtype=np.float64)
    best_paths = np.zeros((sequence_length, number_of_tags), dtype=np.int32)
    best_path = np.zeros(sequence_length, dtype=np.int32)

    if initial_distribution is None:
        # initial_distribution = np.full(number_of_tags, 1.0 / number_of_tags)
        initial_distribution = np.full(number_of_tags, 1.0)

    state_probabilities[0, :] = potentials[0, :] + initial_distribution

    transpose = transition_matrix.transpose()
    for i in range(1, sequence_length):
        prev_p = state_probabilities[i - 1, :] + transpose
        best_previous_nodes = np.argmax(prev_p, axis=1)
        state_probabilities[i] = np.max(prev_p, axis=1)
        state_probabilities[i] += potentials[i, :]
        best_paths[i, :] = best_previous_nodes

    best_path[-1] = np.argmax(state_probabilities[-1, :])
    for i in reversed(range(1, sequence_length)):
        best_path[i - 1] = best_paths[i, best_path[i]]

    return best_path


class NeuralLogDataset(ABC):
    """
    Represents a NeuralLog dataset to train a NeuralLog network.
    """

    program: NeuralLogProgram
    "The NeuralLog program"

    def __init__(self, program, inverse_relations=True):
        """
        Creates a NeuralLogNetwork.

        :param program: the NeuralLog program
        :type program: NeuralLogProgram
        :param inverse_relations: whether the dataset must consider the
        inverse relations
        :type inverse_relations: bool
        """
        self.program = program
        self.inverse_relations = inverse_relations
        self._target_predicates = None

    @property
    def target_predicates(self):
        """
        Gets the target predicates.

        :return: the target predicates
        :rtype: List[Tuple[Predicate, bool]]
        """
        return self._target_predicates

    @target_predicates.setter
    def target_predicates(self, value):
        """
        Sets the target predicates.

        :param value: the target predicates
        :type value: List[Tuple[Predicate, bool]]
        """
        self._target_predicates = value

    @abstractmethod
    def has_example_key(self, key):
        """
        Checks if the dataset contains the example key.

        :param key: the example key
        :type key: Any
        :return: if the dataset contains the atom example
        :rtype: bool
        """
        pass

    @abstractmethod
    def get_dataset(self, example_set=NO_EXAMPLE_SET,
                    batch_size=1, shuffle=False):
        """
        Gets the data set for the example set.

        :param example_set: the name of the example set
        :type example_set: str
        :param batch_size: the batch size
        :type batch_size: int
        :param shuffle: if `True`, shuffles the dataset.
        :type shuffle: bool
        :return: the dataset
        :rtype: tf.data.Dataset
        """
        pass

    @abstractmethod
    def build(self, example_set=NO_EXAMPLE_SET):
        """
        Builds the features and label to train the neural network based on
        the `example_set`.

        :param example_set: the name of the set of examples
        :type example_set: str
        sparse tensor. If `False`, the features are generated as a dense
        tensor of indices, for each index a one hot vector creation is
        necessary.
        :return: the features and labels
        :rtype: (tuple[tf.SparseTensor], tuple[tf.SparseTensor])
        """
        pass

    def get_target_predicates(self):
        """
        Gets a list of tuples containing the target predicates and whether it
        is inverted or not.

        :return: the list of target predicates
        :rtype: list[tuple[Predicate, bool]]
        """
        return self._target_predicates

    @abstractmethod
    def print_predictions(self, model, program, dataset, writer=sys.stdout,
                          dataset_name=None, print_batch_header=False):
        """
        Prints the predictions of `model` to `writer`.

        :param model: the model
        :type model: NeuralLogNetwork
        :param program: the neural program
        :type program: NeuralLogProgram
        :param dataset: the dataset
        :type dataset: tf.data.Dataset
        :param writer: the writer. Default is to write to the standard output
        :type writer: Any
        :param dataset_name: the name of the dataset
        :type dataset_name: str
        :param print_batch_header: if `True`, prints a commented line before
        each batch
        :type print_batch_header: bool
        """
        pass


@neural_log_dataset("default_dataset")
class DefaultDataset(NeuralLogDataset):
    """
    The default NeuralLog dataset.
    """

    def __init__(self, program, inverse_relations=True):
        """
        Creates a DefaultDataset.

        :param program: the NeuralLog program
        :type program: NeuralLogProgram
        :param inverse_relations: whether the dataset must consider the
        inverse relations
        :type inverse_relations: bool
        """
        super(DefaultDataset, self).__init__(program, inverse_relations)
        self._target_predicates = self._compute_target_predicates()
        self.example_keys = self._load_example_keys()

    def _load_example_keys(self):
        example_keys = set()
        for example_set in self.program.examples.values():
            for examples_by_predicate in example_set.values():
                for keys in examples_by_predicate.keys():
                    example_keys.add(keys)
        return example_keys

    def _compute_target_predicates(self):
        target_predicates = []
        predicates = set()
        for example_set in self.program.examples.values():
            for predicate in example_set:
                if predicate in predicates:
                    continue
                predicates.add(predicate)
                target_predicates.append((predicate, False))
                if self.inverse_relations and predicate.arity == 2:
                    target_predicates.append((predicate, True))
        return target_predicates

    # noinspection PyMissingOrEmptyDocstring
    def has_example_key(self, key):
        return key in self.example_keys

    # noinspection PyUnusedLocal,DuplicatedCode
    def call(self, features, labels, *args, **kwargs):
        """
        Used to transform the features and examples from the sparse
        representation to dense in order to train the network.

        :param features: A dense index tensor of the features
        :type features: tuple[tf.SparseTensor]
        :param labels: A tuple sparse tensor of labels
        :type labels: tuple[tf.SparseTensor]
        :param args: additional arguments
        :type args: list
        :param kwargs: additional arguments
        :type kwargs: dict
        :return: the features and label tensors
        :rtype: (tf.Tensor or tuple[tf.Tensor], tuple[tf.Tensor])
        """
        dense_features = []
        count = 0
        for i in range(len(self._target_predicates)):
            predicate, inverted = self._target_predicates[i]
            indices, _ = get_predicate_indices(predicate, inverted)
            for index in indices:
                feature = tf.one_hot(
                    features[count],
                    self.program.get_constant_size(predicate, index))
                dense_features.append(feature)
                count += 1

        labels = tuple(map(lambda x: tf.sparse.to_dense(x), labels))

        if len(dense_features) > 1:
            dense_features = tuple(dense_features)
        else:
            dense_features = dense_features[0]

        # all_dense_features = tuple(all_dense_features)

        return dense_features, labels

    __call__ = call

    # noinspection PyMissingOrEmptyDocstring
    def get_dataset(self, example_set=NO_EXAMPLE_SET,
                    batch_size=1, shuffle=False):
        features, labels = self.build(example_set=example_set)
        # noinspection PyTypeChecker
        if not features:
            return None
        if self.are_features_empty(features):
            return None
        dataset_size = len(features[0])
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            dataset = dataset.shuffle(dataset_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self)
        logger.debug("Dataset %s created with %d example(s)", example_set,
                     dataset_size)
        return dataset

    def are_features_empty(self, features):
        """
        Checks if the features are empty.

        :param features: the features
        :type features: List[List[int]] or Tuple[List[int]]
        :return: `True`, if the features are empty
        :rtype: bool
        """
        size = len(features[0])
        if not size:
            return True
        if size > 1:
            return False
        index = 0
        for i in range(len(self._target_predicates)):
            in_indices, out_index = \
                get_predicate_indices(*self._target_predicates[i])
            for j in range(len(in_indices)):
                empty_value = self._get_out_of_vocabulary_index(
                    self._target_predicates[i][0], in_indices[j])
                if empty_value != features[index][0]:
                    return False
                index += 1

        return True

    def build(self, example_set=NO_EXAMPLE_SET):
        """
        Builds the features and label to train the neural network
        based on
        the `example_set`.

        The labels are always a sparse tensor.

        :param example_set: the name of the set of examples
        :type example_set: str
        sparse tensor. If `False`, the features are generated as a dense
        tensor of indices, for each index a one hot vector creation is
        necessary.
        :return: the features and labels
        :rtype: (tuple[tf.SparseTensor], tuple[tf.SparseTensor]) or
        (list[tuple[tf.SparseTensor]], tuple[tf.SparseTensor])
        """
        examples = self.program.examples.get(example_set, OrderedDict())
        return self._build(examples)

    def ground_atom(self, example):
        """
        Grounds the example by replacing the value of the variables for each
        possible value found in the program.

        :param example: the example
        :type example: Atom
        :return: the grounded atoms
        :rtype: collections.Iterable[Atom]
        """
        if example.is_grounded():
            return example,

        current_atoms = deque([example])
        predicate = example.predicate
        term_types: Tuple[TermType] = self.program.predicates[predicate]
        for i in range(example.arity()):
            if example.terms[i].is_constant():
                continue
            next_atoms = deque()
            for atom in current_atoms:
                if term_types[i].number:
                    terms = list(atom.terms)
                    terms[i] = 0.0
                    next_atoms.append(
                        Atom(predicate, *terms, weight=example.weight))
                else:
                    possible_terms = \
                        self.program.iterable_constants_per_term[(predicate, i)]
                    for constant in possible_terms.values():
                        terms = list(atom.terms)
                        terms[i] = constant
                        next_atoms.append(
                            Atom(predicate, *terms, weight=example.weight))
            current_atoms = next_atoms

        return current_atoms

    def _build(self, examples):
        """
        Builds the features and label to train the neural network based on
        the `example_set`.

        The labels are always a sparse tensor.

        :param examples: the set of examples
        :type examples: Examples
        :return: the features and labels
        :rtype: (tuple[tf.SparseTensor], tuple[tf.SparseTensor])
        """
        output_by_term = OrderedDict()
        input_terms = []
        for predicate, inverted in self._target_predicates:
            facts = examples.get(predicate, dict())
            facts = facts.values()
            for example in facts:
                for fact in self.ground_atom(example):
                    if predicate.arity < 3:
                        input_term = (fact.terms[-1 if inverted else 0],)
                    else:
                        input_term = tuple(fact.terms[0:predicate.arity - 1])
                    if input_term not in output_by_term:
                        output = dict()
                        output_by_term[input_term] = output
                        input_terms.append(input_term)
                    else:
                        output = output_by_term[input_term]

                    if predicate.arity == 1:
                        output[(predicate, inverted)] = fact.weight
                    else:
                        output_term = fact.terms[0 if inverted else -1]
                        # noinspection PyTypeChecker
                        output.setdefault((predicate, inverted), []).append(
                            (output_term, fact.weight))

        all_features = []
        all_labels = []
        for predicate, inverted in self._target_predicates:
            features = [[] for _ in range(max(1, predicate.arity - 1))]
            label_values = []
            label_indices = []

            in_indices, out_index = get_predicate_indices(predicate, inverted)
            for i in range(len(input_terms)):
                outputs = output_by_term[input_terms[i]].get(
                    (predicate, inverted), None)

                constant_index = 0
                for input_index in in_indices:
                    index = None
                    if outputs is not None:
                        index = self.program.get_index_of_constant(
                            predicate, input_index,
                            input_terms[i][constant_index])
                    if index is None:
                        index = self._get_out_of_vocabulary_index(
                            predicate, input_index)
                    features[constant_index].append(index)
                    constant_index += 1

                if outputs is not None:
                    if predicate.arity == 1:
                        label_indices.append([i, 0])
                        label_values.append(outputs)
                    else:
                        # noinspection PyTypeChecker
                        for output_term, output_value in outputs:
                            output_term_index = \
                                self.program.get_index_of_constant(
                                    predicate, out_index, output_term)
                            label_indices.append([i, output_term_index])
                            label_values.append(output_value)

            all_features += features
            if predicate.arity == 1:
                dense_shape = [len(input_terms), 1]
                empty_index = [[0, 0]]
            else:
                dense_shape = [
                    len(input_terms),
                    self.program.get_constant_size(predicate, out_index)]
                empty_index = [[0, 0]]
            if len(label_values) == 0:
                sparse_tensor = tf.SparseTensor(indices=empty_index,
                                                values=[0.0],
                                                dense_shape=dense_shape)
            else:
                sparse_tensor = tf.SparseTensor(indices=label_indices,
                                                values=label_values,
                                                dense_shape=dense_shape)
            sparse_tensor = tf.sparse.reorder(sparse_tensor)
            all_labels.append(sparse_tensor)

        return tuple(all_features), tuple(all_labels)

    def _get_out_of_vocabulary_index(self, predicate, term_index):
        """
        Returns the index of the entity to replace the not found entity.

        :param predicate: the predicate
        :type predicate: Predicate
        :param term_index: the index of the term
        :type term_index: int
        :return: the index of entity to replace the not found one
        :rtype: int
        """
        return -1

    # noinspection PyMissingOrEmptyDocstring
    def print_predictions(self, model, program, dataset, writer=sys.stdout,
                          dataset_name=None, print_batch_header=False):
        return print_neural_log_predictions(
            model, program, self, dataset, writer=writer,
            dataset_name=dataset_name, print_batch_header=print_batch_header)


class AbstractSequenceDataset(DefaultDataset, ABC):
    """
    Represents an Abstract Sequence Dataset.
    """

    def __init__(self, program, pad_index,
                 inverse_relations=False, expand_one_hot=True):
        super().__init__(program, inverse_relations=inverse_relations)
        self.pad_index = pad_index
        self.expand_one_hot = expand_one_hot
        self.example_keys = self._load_example_keys()
        self._output_types = None
        self._output_shapes = None
        self._compute_output_format()

    def _load_example_keys(self):
        example_keys = set()
        for mega_examples in self.program.mega_examples.values():
            for example_set in mega_examples.values():
                for examples_by_predicate in example_set.values():
                    for example in examples_by_predicate:
                        example_keys.add(example.simple_key())
        return example_keys

    def _compute_target_predicates(self):
        target_predicates = []
        predicates = set()
        for mega_examples in self.program.mega_examples.values():
            for example_set in mega_examples.values():
                for predicate in example_set:
                    if predicate in predicates:
                        continue
                    predicates.add(predicate)
                    target_predicates.append((predicate, False))
                    if self.inverse_relations and predicate.arity > 1:
                        target_predicates.append((predicate, True))
        return target_predicates

    def _compute_output_format(self):
        length = 0
        output_types = []
        output_shapes = []
        for predicate, inverted in self._target_predicates:
            length += max(predicate.arity - 1, 1)
            _, index = get_predicate_indices(predicate, inverted)
            size = self.program.get_constant_size(predicate, index)
            output_types.append(tf.float32)
            output_shapes.append((None, size))
        self._output_types = (tf.int32,) + (tuple(output_types),)
        # noinspection PyTypeChecker
        self._output_shapes = ((length, None),) + (tuple(output_shapes),)

    # noinspection PyUnusedLocal,DuplicatedCode
    def call(self, features, labels, *args, **kwargs):
        """
        Used to transform the features and examples from the sparse
        representation to dense in order to train the network.

        :param features: A dense index tensor of the features
        :type features: tuple[tf.SparseTensor]
        :param labels: A tuple sparse tensor of labels
        :type labels: tuple[tf.SparseTensor]
        :param args: additional arguments
        :type args: list
        :param kwargs: additional arguments
        :type kwargs: dict
        :return: the features and label tensors
        :rtype: (tf.Tensor or tuple[tf.Tensor], tuple[tf.Tensor])
        """
        dense_features = []
        count = 0
        for i in range(len(self._target_predicates)):
            predicate, inverted = self._target_predicates[i]
            indices, _ = get_predicate_indices(predicate, inverted)
            for index in indices:
                if self.expand_one_hot:
                    feature = tf.one_hot(
                        features[count],
                        self.program.get_constant_size(predicate, index))
                else:
                    feature = tf.reshape(features[count], [-1, 1])
                dense_features.append(feature)
                count += 1

        if len(dense_features) > 1:
            dense_features = tuple(dense_features)
        else:
            dense_features = dense_features[0]

        return dense_features, labels

    __call__ = call

    # noinspection PyMissingOrEmptyDocstring
    def get_dataset(self, example_set=NO_EXAMPLE_SET,
                    batch_size=1, shuffle=False):
        # noinspection PyTypeChecker
        dataset = tf.data.Dataset.from_generator(
            partial(self.build, example_set),
            output_types=self._output_types,
            output_shapes=self._output_shapes
        )
        dataset_size = len(self.program.mega_examples.get(example_set, []))
        if shuffle:
            dataset = dataset.shuffle(dataset_size)
        dataset = dataset.map(self)
        logger.debug("Dataset %s created with %d example(s)", example_set,
                     dataset_size)
        return dataset

    def build(self, example_set=NO_EXAMPLE_SET):
        """
        Builds the features and label to train the neural network based on
        the `example_set`.

        The labels are always a sparse tensor.

        :param example_set: the name of the set of examples
        :type example_set: str
        sparse tensor. If `False`, the features are generated as a dense
        tensor of indices, for each index a one hot vector creation is
        necessary.
        :return: the features and labels
        :rtype: (tuple[tf.SparseTensor], tuple[tf.SparseTensor])
        """
        mega_examples = self.program.mega_examples.get(
            example_set, OrderedDict())
        for _, examples in sorted(mega_examples.items(), key=lambda x: x[0]):
            features, labels = self._build(examples)
            labels = tuple(map(lambda x: tf.sparse.to_dense(x), labels))
            yield features, labels

    @abstractmethod
    def _build(self, examples):
        """
        Builds the features and label to train the neural network
        based on
        the `example_set`.

        The labels are always a sparse tensor.

        :param examples: the set of examples
        :type examples: dict[Predicate, List[Atom]]
        sparse tensor. If `False`, the features are generated as a dense
        tensor of indices, for each index a one hot vector creation is
        necessary.
        :return: the features and labels
        :rtype: (tuple[list[int]], tuple[tf.SparseTensor])
        """
        pass


@neural_log_dataset("sequence_dataset")
class SequenceDataset(AbstractSequenceDataset):
    """
    The sequence dataset.
    """

    def __init__(self, program, empty_word_index, inverse_relations=True,
                 oov_word="<OOV>", expand_one_hot=True):
        """
        Creates a SequenceDataset.

        :param program: the NeuralLog program
        :type program: NeuralLogProgram
        :param empty_word_index: the index of an entity that is not found in any
        example, to represent an empty entry
        :type empty_word_index: int
        :param inverse_relations: whether the dataset must consider the
        inverse relations
        :type inverse_relations: bool
        :param oov_word: the value to replace out of the vocabulary
        entities
        :type oov_word: str
        :param expand_one_hot: if `True`, expands the indices of the input
        into one hot tensors
        :type expand_one_hot: bool
        """
        super(SequenceDataset, self).__init__(
            program, empty_word_index, inverse_relations, expand_one_hot)
        self.oov_word = get_constant_from_string(oov_word)

    # noinspection DuplicatedCode
    def _build(self, examples):
        all_features = []
        all_label_indices = []
        all_label_values = []
        total_of_rows = 0
        for predicate, inverted in self._target_predicates:
            input_indices, output_index = get_predicate_indices(
                predicate, inverted)
            feature = [[] for _ in range(max(1, predicate.arity - 1))]
            label_indices = []
            label_values = []
            facts = examples.get(predicate, [])
            for example in facts:
                for fact in self.ground_atom(example):
                    if predicate.arity < 3:
                        input_terms = (fact.terms[-1 if inverted else 0],)
                    else:
                        input_terms = tuple(fact.terms[0:predicate.arity - 1])

                    count = 0
                    for input_index, input_term in zip(input_indices,
                                                       input_terms):
                        input_value = self.program.get_index_of_constant(
                            predicate, input_index, input_term)
                        if input_value is None:
                            input_value = self._get_out_of_vocabulary_index(
                                predicate, input_index)
                        feature[count].append(input_value)
                        count += 1

                    if predicate.arity == 1:
                        label_indices.append([total_of_rows, 0])
                    else:
                        output_term = fact.terms[output_index]
                        output_value = self.program.get_index_of_constant(
                            predicate, output_index, output_term)
                        label_indices.append([total_of_rows, output_value])
                    label_values.append(fact.weight)
                    total_of_rows += 1
            all_label_indices.append(label_indices)
            all_label_values.append(label_values)
            all_features += feature

        all_labels = []
        examples_offset = 0
        features_offset = 0
        for i in range(len(self._target_predicates)):
            # Features
            number_of_features = max(self._target_predicates[i][0].arity - 1, 1)
            length = len(all_features[features_offset])
            for j in range(number_of_features):
                all_features[features_offset + j] = \
                    ([self.pad_index] * examples_offset) + \
                    all_features[features_offset + j]
                all_features[features_offset + j] += \
                    [self.pad_index] * (
                            total_of_rows - examples_offset - length)
            examples_offset += length
            features_offset += number_of_features

            # Labels
            predicate, index = self._target_predicates[i]
            _, output_index = get_predicate_indices(predicate, index)
            if predicate.arity == 1:
                dense_shape = [total_of_rows, 1]
                empty_index = [[0, 0]]
            else:
                dense_shape = [
                    total_of_rows,
                    self.program.get_constant_size(predicate, output_index)]
                empty_index = [[0, 0]]
            if len(all_label_values[i]) == 0:
                sparse_tensor = tf.SparseTensor(
                    indices=empty_index, values=[0.0], dense_shape=dense_shape)
            else:
                sparse_tensor = tf.SparseTensor(
                    indices=all_label_indices[i], values=all_label_values[i],
                    dense_shape=dense_shape)
            all_labels.append(sparse_tensor)

        return tuple(all_features), tuple(all_labels)

    def _get_out_of_vocabulary_index(self, predicate, term_index):
        """
        Returns the index of the entity to replace the not found entity.

        :param predicate: the predicate
        :type predicate: Predicate
        :param term_index: the index of the term
        :type term_index: int
        :return: the index of entity to replace the not found one
        :rtype: int
        """
        return self.program.get_index_of_constant(predicate, term_index,
                                                  self.oov_word)


def _atom_processor(predicate, feature, label, weight):
    return str(AtomClause(Atom(predicate, feature, label, weight=weight)))


# noinspection PyUnusedLocal
def _conll_processor(predicate, feature, label, weight):
    return f"{feature.value}\t{label.value}"


@neural_log_dataset("language_dataset")
class LanguageDataset(AbstractSequenceDataset):
    """
    Class to process mega examples as phrases.
    """

    def __init__(self,
                 program,
                 inverse_relations,
                 vocabulary_file,
                 initial_token="[CLS]",
                 final_token="[SEP]",
                 pad_token="[PAD]",
                 sub_token_label="X",
                 maximum_sentence_length=None,
                 do_lower_case=False,
                 pad_to_maximum_length=False,
                 string_processor=None
                 # expand_one_hot=False,
                 # language_predicates=None,
                 ):

        self.tokenizer = FullTokenizer(
            vocabulary_file, do_lower_case=do_lower_case)

        self.maximum_sentence_length = maximum_sentence_length

        self.initial_token = Constant(initial_token)
        self.initial_token_index = self.tokenizer.vocab[initial_token]

        self.final_token = Constant(final_token)
        self.final_token_index = self.tokenizer.vocab[final_token]

        self.skip_tokens = [initial_token, final_token]

        self.pad_token = Constant(pad_token)
        self.pad_index = self.tokenizer.vocab[pad_token]

        self.sub_token_label = Constant(sub_token_label)

        self.pad_to_maximum_length = \
            pad_to_maximum_length and maximum_sentence_length is not None

        super(LanguageDataset, self).__init__(
            program, self.pad_index, inverse_relations, expand_one_hot=False)

        if string_processor is not None and string_processor.lower() == 'conll':
            self._string_processor = _conll_processor
        else:
            self._string_processor = _atom_processor

    # noinspection PyMissingOrEmptyDocstring
    def call(self, features, labels, *args, **kwargs):
        dense_features = []
        count = 0
        for i in range(len(self._target_predicates)):
            predicate, inverted = self._target_predicates[i]
            indices, _ = get_predicate_indices(predicate, inverted)
            for index in indices:
                if self.expand_one_hot:
                    feature = tf.one_hot(
                        features[count],
                        self.program.get_constant_size(predicate, index))
                else:
                    feature = features[count]
                dense_features.append(feature)
                count += 1

        if len(dense_features) > 1:
            dense_features = tuple(dense_features)
        else:
            dense_features = dense_features[0]

        return dense_features, labels

    __call__ = call

    def _compute_output_format(self):
        batch_size = None
        if self.pad_to_maximum_length:
            batch_size = self.maximum_sentence_length
        length = 0
        output_types = []
        output_shapes = []
        for predicate, inverted in self._target_predicates:
            length += max(predicate.arity - 1, 1)
            _, index = get_predicate_indices(predicate, inverted)
            size = self.program.get_constant_size(predicate, index)
            output_types.append(tf.float32)
            output_shapes.append((batch_size, size))
        self._output_types = (tf.int32,) + (tuple(output_types),)
        # noinspection PyTypeChecker
        self._output_shapes = ((length, batch_size),) + (tuple(output_shapes),)

    # noinspection PyMissingOrEmptyDocstring
    def get_dataset(self, example_set=NO_EXAMPLE_SET,
                    batch_size=1, shuffle=False):
        dataset = super().get_dataset(
            example_set, batch_size=batch_size, shuffle=shuffle)
        return dataset.batch(batch_size)

    def _reached_maximum_length(self, current_length):
        if self.maximum_sentence_length is None:
            return False
        return current_length > self.maximum_sentence_length - 2

    # noinspection DuplicatedCode
    def _build(self, examples):
        all_features, all_label_indices, all_label_values, total_of_rows = \
            self._preprocess_examples(examples)

        all_labels = []
        examples_offset = 0
        features_offset = 0
        for i in range(len(self._target_predicates)):
            # Features
            number_of_features = max(
                self._target_predicates[i][0].arity - 1, 1)
            length = len(all_features[features_offset])
            for j in range(number_of_features):
                all_features[features_offset + j] = \
                    ([self.pad_index] * examples_offset) + \
                    all_features[features_offset + j]
                all_features[features_offset + j] += \
                    [self.pad_index] * (
                            total_of_rows - examples_offset - length)
            examples_offset += length
            features_offset += number_of_features

            # Labels
            predicate, index = self._target_predicates[i]
            _, output_index = get_predicate_indices(predicate, index)
            if predicate.arity == 1:
                dense_shape = [total_of_rows, 1]
                empty_index = [[0, 0]]
            else:
                dense_shape = [
                    total_of_rows,
                    self.program.get_constant_size(predicate, output_index)]
                empty_index = [[0, 0]]
            if len(all_label_values[i]) == 0:
                sparse_tensor = tf.SparseTensor(
                    indices=empty_index, values=[0.0],
                    dense_shape=dense_shape)
            else:
                sparse_tensor = tf.SparseTensor(
                    indices=all_label_indices[i],
                    values=all_label_values[i],
                    dense_shape=dense_shape)
            all_labels.append(sparse_tensor)

        return tuple(all_features), tuple(all_labels)

    # noinspection DuplicatedCode
    def _preprocess_examples(self, examples):
        all_features = []
        all_label_indices = []
        all_label_values = []
        total_of_rows = 0
        for predicate, inverted in self._target_predicates:
            input_indices, output_index = get_predicate_indices(
                predicate, inverted)
            feature = [[] for _ in range(max(1, predicate.arity - 1))]
            label_indices = []
            label_values = []
            facts = examples.get(predicate, [])

            # Adds the initial token
            for i in range(len(input_indices)):
                feature[i].append(self.initial_token_index)
            if predicate.arity == 1:
                label_indices.append([total_of_rows, 0])
            else:
                index = self.program.get_index_of_constant(
                    predicate, -1, self.initial_token)
                label_indices.append([total_of_rows, index])
            label_values.append(1.0)
            total_of_rows += 1

            reached_maximum_length = False
            for example in facts:
                for fact in self.ground_atom(example):
                    if predicate.arity < 3:
                        input_terms = (fact.terms[-1 if inverted else 0],)
                    else:
                        input_terms = tuple(
                            fact.terms[0:predicate.arity - 1])

                    count = 0
                    input_values = []
                    maximum_rows = \
                        self.maximum_sentence_length - total_of_rows - 1
                    for input_index, input_term in zip(input_indices,
                                                       input_terms):
                        input_values = self.tokenizer.tokenize(input_term.value)
                        if input_values is None:
                            input_values = [self._get_out_of_vocabulary_index(
                                predicate, input_index)]
                        else:
                            input_values = self.tokenizer.convert_tokens_to_ids(
                                input_values)
                        feature[count].extend(input_values[:maximum_rows])
                        count += 1

                    if predicate.arity == 1:
                        label_indices.append([total_of_rows, 0])
                    else:
                        output_term = fact.terms[output_index]
                        output_value = self.program.get_index_of_constant(
                            predicate, output_index, output_term)
                        label_indices.append([total_of_rows, output_value])
                    label_values.append(fact.weight)
                    total_of_rows += 1

                    if self._reached_maximum_length(total_of_rows):
                        reached_maximum_length = True
                        break

                    sub_token_index = self.program.get_index_of_constant(
                        predicate, -1, self.sub_token_label)
                    # Adds the sub_token labels
                    for _ in range(len(input_values) - 1):
                        if predicate.arity == 1:
                            label_indices.append([total_of_rows, 0])
                        else:
                            label_indices.append(
                                [total_of_rows, sub_token_index])
                        label_values.append(1.0)
                        total_of_rows += 1
                        if self._reached_maximum_length(total_of_rows):
                            reached_maximum_length = True
                            break

                if reached_maximum_length:
                    break

            # Adds the final token
            for i in range(len(input_indices)):
                feature[i].append(self.final_token_index)
            if predicate.arity == 1:
                label_indices.append([total_of_rows, 0])
            else:
                index = self.program.get_index_of_constant(
                    predicate, -1, self.final_token)
                label_indices.append([total_of_rows, index])
            label_values.append(1.0)
            total_of_rows += 1

            # Adds pads
            if self.pad_to_maximum_length:
                index = self.program.get_index_of_constant(
                    predicate, -1, self.pad_token)
                for _ in range(self.maximum_sentence_length - total_of_rows):
                    for i in range(len(input_indices)):
                        feature[i].append(self.pad_index)
                    if predicate.arity == 1:
                        label_indices.append([total_of_rows, 0])
                    else:
                        label_indices.append([total_of_rows, index])
                    label_values.append(1.0)
                    total_of_rows += 1

            all_label_indices.append(label_indices)
            all_label_values.append(label_values)
            all_features += feature

        return all_features, all_label_indices, all_label_values, total_of_rows

    # noinspection PyMissingOrEmptyDocstring,DuplicatedCode
    def print_predictions(self, model, program, dataset,
                          writer=sys.stdout, dataset_name=None,
                          print_batch_header=False):
        prefix_length = len(PARTIAL_WORD_PREFIX)

        count = 0
        batches = None
        if print_batch_header and dataset_name is not None:
            batches = list(program.mega_examples[dataset_name].keys())

        for features, _ in dataset:
            if print_batch_header:
                batch = batches[count] if batches is not None else count
                print("%% Batch:", batch, file=writer, sep="\t")
                count += 1
            y_scores = model.predict(features)
            if len(model.predicates) == 1:
                # if not isinstance(neural_dataset, WordCharDataset):
                y_scores = [y_scores]
                if model.predicates[0][0].arity < 3:
                    features = [features]
            # i iterates over the predicates
            for i in range(len(model.predicates)):
                predicate, inverted = model.predicates[i]
                if inverted:
                    continue
                row_scores = y_scores[i]
                # j iterates over the mega examples
                for j in range(len(row_scores)):
                    y_score = row_scores[j]
                    offset = sum(model.input_sizes[:i])
                    x_k = features[offset][j].numpy()
                    subjects = self.tokenizer.convert_ids_to_tokens(x_k)

                    processed_token = None
                    label_index = None
                    for index, sub in enumerate(subjects):
                        if sub == self.pad_token.value:
                            break
                        if sub.startswith(PARTIAL_WORD_PREFIX):
                            processed_token += sub[prefix_length:]
                        else:
                            if processed_token is not None:
                                max_label = y_score[label_index].argmax()
                                obj = program.get_constant_by_index(
                                    predicate, -1, max_label)
                                weight = float(y_score[label_index][max_label])
                                feature = Quote(f'"{processed_token}"')
                                string_format = self._string_processor(
                                    predicate, feature, obj, weight)
                                print(string_format, file=writer)
                            if sub in self.skip_tokens:
                                continue
                            processed_token = sub
                            label_index = index
                    print(file=writer)


@neural_log_dataset("word_char_dataset")
class WordCharDataset(SequenceDataset):
    """
    Class to represent a Word and Char dataset.

    This class considers ternary predicates as being composed by an word,
    a sequence of characters (represented as a string) and a a class.

    The word and the class will be treated as usual. The sequence of
    characters will be transformed into a vector of index of the character
    entity in a given predicate. The vector will have the size of the
    largest sequence in the batch.
    """

    def __init__(self, program, empty_word_index, empty_char_index,
                 character_predicate, character_predicate_index=0,
                 inverse_relations=True, oov_word="<OOV>", oov_char="<OOV>",
                 expand_one_hot=True, char_pad=0):
        """
        Creates a word char dataset.

        :param program: the NeuralLog program
        :type program: NeuralLogProgram
        :param empty_word_index: the index of an entity that is not found in any
        example, to represent an empty entry
        :type empty_word_index: int
        :param character_predicate: the predicate to get the index of the
        characters
        :type character_predicate: str
        :param character_predicate_index: the index of the term in the character
        predicate, to get the index of the characters
        :type character_predicate_index: int
        :param inverse_relations: whether the dataset must consider the
        inverse relations
        :type inverse_relations: bool
        :param oov_word: the value to replace out of the vocabulary words
        :type oov_word: str
        :param oov_char: the value to replace out of the vocabulary chars
        :type oov_char: str
        :param expand_one_hot: if `True`, expands the indices of the input
        into one hot tensors
        :type expand_one_hot: bool
        :param char_pad: the number of empty elements to append at the end of
        the char sequence
        :type char_pad: int
        """
        super(WordCharDataset, self).__init__(
            program, empty_word_index, inverse_relations,
            oov_word, expand_one_hot)
        self.empty_char_index = empty_char_index
        self.character_predicate = \
            get_predicate_from_string(character_predicate)
        self.character_predicate_index = character_predicate_index
        self.oov_char = get_constant_from_string(oov_char)
        self._ooc_char_index = self._get_out_of_vocabulary_index(
            get_predicate_from_string(character_predicate),
            character_predicate_index
        )
        self.char_pad = max(char_pad, 0)

    # noinspection PyMissingOrEmptyDocstring
    def has_example_key(self, key):
        return True

    def _compute_target_predicates(self):
        target_predicates = []
        predicates = set()
        for mega_examples in self.program.mega_examples.values():
            for example_set in mega_examples.values():
                for predicate in example_set:
                    if predicate in predicates:
                        continue
                    predicates.add(predicate)
                    predicate = Predicate(predicate.name, predicate.arity + 1)
                    self.program.logic_predicates.add(predicate)
                    target_predicates.append((predicate, False))
        return target_predicates

    def _compute_output_format(self):
        length = 0
        label_types = []
        label_shapes = []
        feature_shapes = []
        for predicate, inverted in self._target_predicates:
            length += max(predicate.arity - 1, 1)
            _, index = get_predicate_indices(predicate, inverted)
            size = self.program.get_constant_size(predicate, index)
            label_types.append(tf.float32)
            label_shapes.append((None, size))
            if length == 1:
                feature_shapes.append((None, 1))
            else:
                feature_shapes += [(None, 1)] * (length - 1)
                feature_shapes.append((None, None))
        feature_types = (tf.int32, tf.int32)
        self._output_types = (feature_types, tuple(label_types))
        # noinspection PyTypeChecker
        self._output_shapes = (tuple(feature_shapes), tuple(label_shapes))

    # noinspection PyMissingOrEmptyDocstring
    def get_dataset(self, example_set=NO_EXAMPLE_SET,
                    batch_size=1, shuffle=False):
        # noinspection PyTypeChecker
        dataset = tf.data.Dataset.from_generator(
            partial(self.build, example_set),
            output_types=self._output_types,
            output_shapes=self._output_shapes
        )
        dataset_size = len(self.program.mega_examples.get(example_set, []))
        if shuffle:
            dataset = dataset.shuffle(dataset_size)

        logger.debug("Dataset %s created with %d example(s)", example_set,
                     dataset_size)
        return dataset

    def call(self, features, labels, *args, **kwargs):
        """
        Used to transform the features and examples from the sparse
        representation to dense in order to train the network.

        :param features: A dense index tensor of the features
        :type features: tuple[tf.SparseTensor]
        :param labels: A tuple sparse tensor of labels
        :type labels: tuple[tf.SparseTensor]
        :param args: additional arguments
        :type args: list
        :param kwargs: additional arguments
        :type kwargs: dict
        :return: the features and label tensors
        :rtype: (tf.Tensor or tuple[tf.Tensor], tuple[tf.Tensor])
        """
        dense_features = []
        count = 0
        for i in range(len(self._target_predicates)):
            predicate, inverted = self._target_predicates[i]
            indices, _ = get_predicate_indices(predicate, inverted)
            for index in indices:
                if self.expand_one_hot:
                    feature = tf.one_hot(
                        features[count],
                        self.program.get_constant_size(predicate, index))
                else:
                    feature = tf.constant(features[count])
                dense_features.append(feature)
                count += 1

        if len(dense_features) > 1:
            dense_features = tuple(dense_features)
        else:
            dense_features = dense_features[0]

        if len(labels) == 1:
            labels = labels[0]

        return dense_features, labels

    __call__ = call

    # noinspection DuplicatedCode
    def _build(self, examples):
        """
        Builds the features and label to train the neural network based on
        the `example_set`.

        The labels are always a sparse tensor.

        :param examples: the set of examples
        :type examples: dict[Predicate, List[Atom]]
        sparse tensor. If `False`, the features are generated as a dense
        tensor of indices, for each index a one hot vector creation is
        necessary.
        :return: the features and labels
        :rtype: (tuple[list[int]], tuple[tf.SparseTensor])
        """
        all_features = []  # type: List[int] or List[List[int]]
        all_label_indices = []
        all_label_values = []
        row_index = 0
        max_lengths = []
        for predicate, inverted in self._target_predicates:
            input_indices, output_index = get_predicate_indices(
                predicate, inverted)
            output_index -= 1
            real_predicate = Predicate(predicate.name, predicate.arity - 1)
            feature = [[] for _ in range(max(1, predicate.arity - 1))]
            label_indices = []
            label_values = []
            facts = examples.get(real_predicate, [])
            max_length = -1

            for example in facts:
                for fact in self.ground_atom(example):
                    input_terms = tuple(fact.terms[0:predicate.arity - 2])

                    count = 0
                    for input_index, in_term in zip(input_indices, input_terms):
                        in_term = get_term_from_string(str(in_term).lower())
                        input_value = self.program.get_index_of_constant(
                            real_predicate, input_index, in_term)
                        if input_value is None:
                            input_value = self._get_out_of_vocabulary_index(
                                real_predicate, input_index)
                        feature[count].append([input_value])
                        count += 1

                    if predicate.arity > 2:
                        char_features = []
                        last_term = input_terms[-1].value
                        max_length = max(len(last_term), max_length)
                        for char in last_term:
                            input_value = self.program.get_index_of_constant(
                                self.character_predicate,
                                self.character_predicate_index,
                                get_constant_from_string(char)
                            )
                            if input_value is None:
                                input_value = self._ooc_char_index
                            char_features.append(input_value)
                        feature[-1].append(char_features)

                    output_term = fact.terms[output_index]
                    output_value = self.program.get_index_of_constant(
                        real_predicate, output_index, output_term)
                    label_indices.append([row_index, output_value])

                    label_values.append(fact.weight)
                    row_index += 1
            max_lengths.append(max_length + self.char_pad)
            all_label_indices.append(label_indices)
            all_label_values.append(label_values)
            all_features += feature

        all_labels = []
        examples_offset = 0
        features_offset = 0
        for i in range(len(self._target_predicates)):
            # Features
            arity = self._target_predicates[i][0].arity
            number_of_features = max(arity - 1, 1)
            length = len(all_features[features_offset])
            if arity > 2:
                number_of_features -= 1
            for j in range(number_of_features):
                all_features[features_offset + j] = \
                    ([self.empty_word_index] * examples_offset) + \
                    all_features[features_offset + j]
                all_features[features_offset + j] += \
                    [self.empty_word_index] * (
                            row_index - examples_offset - length)
            if arity > 2:
                j = number_of_features
                adjusted_features = []
                for current in all_features[features_offset + j]:
                    # noinspection PyTypeChecker
                    adjusted_features.append(
                        current +
                        ([self.empty_char_index] *
                         (max_lengths[i] - len(current))))
                all_features[features_offset + j] = \
                    ([[self.empty_char_index] * max_lengths[i]] *
                     examples_offset) + adjusted_features
                all_features[features_offset + j] += \
                    [[self.empty_char_index] * max_lengths[i]] * (
                            row_index - examples_offset - length)
                number_of_features += 1
            examples_offset += length
            features_offset += number_of_features

            # Labels
            predicate, index = self._target_predicates[i]
            real_predicate = Predicate(predicate.name, predicate.arity - 1)
            _, output_index = get_predicate_indices(predicate, index)
            output_index -= 1
            if predicate.arity == 1:
                dense_shape = [row_index, 1]
                empty_index = [[0, 0]]
            else:
                dense_shape = [
                    row_index,
                    self.program.get_constant_size(
                        real_predicate, output_index)]
                empty_index = [[0, 0]]
            if len(all_label_values[i]) == 0:
                sparse_tensor = tf.SparseTensor(
                    indices=empty_index, values=[0.0], dense_shape=dense_shape)
            else:
                sparse_tensor = tf.SparseTensor(
                    indices=all_label_indices[i], values=all_label_values[i],
                    dense_shape=dense_shape)
            all_labels.append(sparse_tensor)

        return tuple(all_features), tuple(all_labels)

    # noinspection PyMissingOrEmptyDocstring
    def print_predictions(self, model, program, dataset, writer=sys.stdout,
                          dataset_name=None, print_batch_header=False):
        return _print_word_char_predictions(
            model, program, self, dataset, writer=writer,
            dataset_name=dataset_name, print_batch_header=print_batch_header)
