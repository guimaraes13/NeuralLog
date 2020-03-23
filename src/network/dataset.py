"""
Handles the examples.
"""
import logging
import sys
from collections import OrderedDict
from functools import partial

import numpy as np
import tensorflow as tf

from src.knowledge.program import NeuralLogProgram, NO_EXAMPLE_SET
from src.language.language import AtomClause, Atom, Predicate, \
    get_constant_from_string
from src.network import registry

logger = logging.getLogger()

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


# noinspection PyTypeChecker
def print_neural_log_predictions(model, neural_program, dataset,
                                 writer=sys.stdout, dataset_name=None,
                                 print_batch_header=False):
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
    :param dataset_name: the name of the dataset
    :type dataset_name: str
    :param print_batch_header: if `True`, prints a commented line before each
    batch
    :type print_batch_header: bool
    """
    neural_dataset = model.dataset  # type: NeuralLogDataset
    count = 0
    for features, _ in dataset:
        if print_batch_header:
            print("%% Batch:", count, file=writer, sep="\t")
            count += 1
        y_scores = model.predict(features)
        if len(model.predicates) == 1:
            y_scores = [y_scores]
            features = [features]
        for i in range(len(model.predicates)):
            predicate, inverted = model.predicates[i]
            if inverted:
                continue
            for feature, y_score in zip(features[i], y_scores[i]):
                x = feature.numpy()
                if len(x.shape) == 1:
                    subject_index = x[0]
                    if subject_index < 0:
                        continue
                else:
                    if x.max() == 0.0:
                        continue
                    subject_index = np.argmax(x)
                subject = neural_program.get_constant_by_index(
                    predicate, 0, subject_index)
                if predicate.arity == 1:
                    clause = AtomClause(Atom(predicate, subject,
                                             weight=float(y_score)))
                    print(clause, file=writer)
                else:
                    clauses = []
                    for index in range(len(y_score)):
                        object_term = neural_program.get_constant_by_index(
                            predicate, 1, index)
                        prediction = Atom(predicate, subject, object_term,
                                          weight=float(y_score[index]))
                        if dataset_name is not None and \
                                not neural_dataset.has_example_key(
                                    prediction.simple_key()):
                            continue
                        clauses.append(AtomClause(prediction))

                    if len(clauses) > 0:
                        clause = AtomClause(Atom(predicate, subject, "X"))
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


class NeuralLogDataset:
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

    def has_example_key(self, key):
        """
        Checks if the dataset contains the example key.

        :param key: the example key
        :type key: Any
        :return: if the dataset contains the atom example
        :rtype: bool
        """
        pass

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
        pass


# TODO: adjust dataset to predicates with higher arity
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
                if self.inverse_relations and predicate.arity > 1:
                    target_predicates.append((predicate, True))
        return target_predicates

    # noinspection PyMissingOrEmptyDocstring
    def get_target_predicates(self):
        return self._target_predicates

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
        if len(self._target_predicates) == 1:
            features = [features]
        dense_features = []
        for i in range(len(self._target_predicates)):
            predicate, inverted = self._target_predicates[i]
            indices, _ = get_predicate_indices(predicate, inverted)
            count = 0
            for index in indices:
                feature = tf.one_hot(
                    features[i][count],
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
        if len(self._target_predicates) == 1:
            dataset_size = len(features[0])
        else:
            dataset_size = len(features[0][0])
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            dataset = dataset.shuffle(dataset_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self)
        logger.info("Dataset %s created with %d example(s)", example_set,
                    dataset_size)
        return dataset

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

    def _build(self, examples):
        """
        Builds the features and label to train the neural network based on
        the `example_set`.

        The labels are always a sparse tensor.

        :param examples: the set of examples
        :type examples: Dict[Predicate, Dict[Any, Atom]]
        sparse tensor. If `False`, the features are generated as a dense
        tensor of indices, for each index a one hot vector creation is
        necessary.
        :return: the features and labels
        :rtype: (tuple[tf.SparseTensor], tuple[tf.SparseTensor])
        """
        output_by_term = OrderedDict()
        input_terms = []
        for predicate, inverted in self._target_predicates:
            facts = examples.get(predicate, dict())
            facts = facts.values()
            for fact in facts:
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
                constant_index = 0
                for input_index in in_indices:
                    index = self.program.get_index_of_constant(
                        predicate, input_index, input_terms[i][constant_index])
                    if index is None:
                        index = self._get_out_of_vocabulary_index(
                            predicate, input_index)
                    features[constant_index].append(index)
                    constant_index += 1

                outputs = output_by_term[input_terms[i]].get(
                    (predicate, inverted), None)
                if outputs is not None:
                    if predicate.arity == 1:
                        label_indices.append([i, 0])
                        label_values.append(outputs)
                    else:
                        for output_term, output_value in outputs:
                            output_term_index = \
                                self.program.get_index_of_constant(
                                    predicate, out_index, output_term)
                            label_indices.append([i, output_term_index])
                            label_values.append(output_value)

            all_features.append(features)
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

        if len(self._target_predicates) == 1:
            all_features = all_features[0]

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


@neural_log_dataset("sequence_dataset")
class SequenceDataset(DefaultDataset):
    """
    The sequence dataset.
    """

    def __init__(self, program, empty_entry, inverse_relations=True,
                 out_of_vocabulary="<OOV>", expand_one_hot=True):
        """
        Creates a SequenceDataset.

        :param program: the NeuralLog program
        :type program: NeuralLogProgram
        :param empty_entry: the index of an entity that is not found in any
        example, to represent an empty entry
        :type empty_entry: int
        :param inverse_relations: whether the dataset must consider the
        inverse relations
        :type inverse_relations: bool
        :param out_of_vocabulary: the value to replace out of the vocabulary
        entities
        :type out_of_vocabulary: Term
        :param expand_one_hot: if `True`, expands the indices of the input
        into one hot tensors
        :type expand_one_hot: bool
        """
        super(SequenceDataset, self).__init__(program, inverse_relations)
        self.empty_entry = empty_entry
        self.out_of_vocabulary = get_constant_from_string(out_of_vocabulary)
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
        length = len(self._target_predicates)
        output_types = []
        output_shapes = []
        for predicate, inverted in self._target_predicates:
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
        for i in range(len(self._target_predicates)):
            predicate, inverted = self._target_predicates[i]
            index, _ = get_predicate_indices(predicate, inverted)
            if self.expand_one_hot:
                feature = tf.one_hot(
                    features[i],
                    self.program.get_constant_size(predicate, index))
            else:
                feature = tf.reshape(features[i], [-1, 1])
            dense_features.append(feature)

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
        logger.info("Dataset %s created with %d example(s)", example_set,
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

    def _build(self, examples, not_found_value=0):
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
        all_features = []
        all_label_indices = []
        all_label_values = []
        row_index = 0
        for predicate, inverted in self._target_predicates:
            input_index, output_index = get_predicate_indices(
                predicate, inverted)
            feature = []
            label_indices = []
            label_values = []
            facts = examples.get(predicate, [])
            for fact in facts:
                input_term = fact.terms[input_index]
                input_value = self.program.get_index_of_constant(
                    predicate, input_index, input_term)
                if input_value is None:
                    input_value = self._get_out_of_vocabulary_index(
                        predicate, input_index)
                feature.append(input_value)
                if predicate.arity == 1:
                    label_indices.append([row_index, 0])
                else:
                    output_term = fact.terms[output_index]
                    output_value = self.program.get_index_of_constant(
                        predicate, output_index, output_term)
                    label_indices.append([row_index, output_value])
                label_values.append(fact.weight)
                row_index += 1
            all_label_indices.append(label_indices)
            all_label_values.append(label_values)
            all_features.append(feature)

        all_labels = []
        offset = 0
        for i in range(len(self._target_predicates)):
            all_features[i] = \
                ([self.empty_entry] * offset) + all_features[i]
            length = len(all_features[i])
            all_features[i] += \
                [self.empty_entry] * (row_index - offset - length)
            offset += length

            predicate, index = self._target_predicates[i]
            _, output_index = get_predicate_indices(predicate, index)
            if predicate.arity == 1:
                dense_shape = [row_index, 1]
                empty_index = [[0, 0]]
            else:
                dense_shape = [
                    row_index,
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
                                                  self.out_of_vocabulary)
