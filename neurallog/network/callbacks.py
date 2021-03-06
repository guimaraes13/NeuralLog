#  Copyright 2021 Victor Guimarães
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
File to store useful callback implementations.
"""

import logging
import re
from typing import Any, List, Dict, Set, Tuple

import numpy as np
import tensorflow.keras.callbacks as keras_callbacks
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback

from neurallog.knowledge.program import NeuralLogProgram
from neurallog.language.language import Predicate, Atom
from neurallog.network import registry
from neurallog.network.network import NeuralLogNetwork
from neurallog.run.command import TRAIN_SET_NAME, VALIDATION_SET_NAME, \
    TEST_SET_NAME

FALSE_VALUES = ["false", "no", "n", 0, 0.0]

METRIC_LOG_PATTERN = "\t{:.5f}:\t{}\n"
OUTPUT_MATCH = re.compile(r"output_[0-9]+")

MEAN_RANK_METRIC_FORMAT = "mean_rank_{}"
MEAN_RECIPROCAL_RANK_METRIC_FORMAT = "mean_reciprocal_rank_{}"
TOP_K_RANK_METRIC_FORMAT = "top_k_rank_{}"

LOG_FORMAT = "output_{}_{}"

logger = logging.getLogger(__name__)

callbacks = dict()


def neural_log_callback(identifier):
    """
    A decorator for NeuralLog callbacks.

    :param identifier: the identifier of the callback
    :type identifier: str
    :return: the decorated function
    :rtype: function
    """
    return lambda x: registry(x, identifier, callbacks)


def get_neural_log_callback(identifier):
    """
    Gets the callback from `identifier`.

    :param identifier: the identifier
    :type identifier: str
    :raise ValueError: if the callback is not found
    :return: the callback
    :rtype: type
    """
    callback = callbacks.get(identifier, None)
    if callback is None:
        callback = getattr(keras_callbacks, identifier, None)
    return callback


def get_bigger_and_equals_counts(y_score, object_index, filtered_objects):
    """
    Gets the number of objects that have a score bigger than and equal to the
    score of the `object_index`.

    Filtering the objects whose index are in `filtered_objects`.

    :param y_score: an array with the score of each object
    :type y_score: np.ndarray, np.array
    :param object_index: the index of the current object
    :type object_index: int
    :param filtered_objects: a (possible empty) set with the indices of the
    objects to be ignored in the rank count
    :type filtered_objects: set[int]
    :return: the number of objects whose the scores are bigger than the current
    object and the number of objects whose the scores are equal to the current
    object
    :rtype: (int, int)
    """
    object_score = y_score[object_index]
    bigger_count = 0
    equals_count = 0

    for j in range(len(y_score)):
        if j == object_index or j in filtered_objects:
            continue
        if y_score[j] > object_score:
            bigger_count += 1
        if y_score[j] == object_score:
            equals_count += 1
    return bigger_count, equals_count


def get_rank_function(rank_method):
    """
    Gets the function to calculate the rank of a entity based on the
    `rank_method`. The function receives two inputs: (1) the number of entities
    whose rank is greater than the rank of the target entity; (2) the number
    of entities whose rank is equal to the rank of the target entity
    (without the target entity itself).

    :param rank_method: the rank method. If `pessimistic`, sums the number of
    entities with rank greater than or equals to the rank of the target
    entity; if `average`, sums the number of entities with rank greater than
    the rank of the target entities and half the number of entities with rank
    equals to the rank of the target entity; otherwise, returns the `optimistic`
    rank, which sums only the number of entities with rank greater than the
    rank of the target entity.
    :type rank_method: str
    :return: the function
    :rtype: function
    """
    if rank_method == "pessimistic":
        return lambda bigger_than, equals_to: bigger_than + equals_to + 1
    elif rank_method == "average":
        return lambda bigger_than, equals_to: \
            bigger_than + int(equals_to / 2) + 1
    else:
        return lambda bigger_than, equals_to: bigger_than + 1


def get_formatted_name(name, output_map):
    """
    Gets the formatted output name.

    :param name: the output name
    :type name: str
    :param output_map: the map of the outputs of the neural network by the
    predicate
    :type output_map: dict[str, tuple(Predicate, bool)] or None
    :return: the formatted name or `name`
    :rtype: str
    """
    if output_map is None:
        return name

    match = OUTPUT_MATCH.match(name)
    if match is not None:
        suffix = name[match.end():]
        output_key = match.group()
        predicate, inverted = output_map.get(output_key,
                                             (output_key, False))
        name = str(predicate).replace("/", "_")
        name += "_inv" if inverted else ""
        name += suffix
    return name


class EpochLogger(Callback):
    """
    Callback to log the measures of the model after each epoch.
    """

    def __init__(self, number_of_epochs, output_map=None):
        """
        Callback to log the measures of the model after each epoch.

        :param number_of_epochs: the total number of epochs of the training
        :type number_of_epochs: int or None
        :param output_map: the map of the outputs of the neural network by the
        predicate
        :type output_map: dict[str, tuple(Predicate, bool)] or None
        """
        super(Callback, self).__init__()
        self.number_of_epochs = str(number_of_epochs) or "Unknown"
        self.output_map = output_map

    def format_metrics_for_epoch(self, metrics, epoch=None):
        """
        Formats the metrics on `metrics` for the given epoch.

        :param metrics: a dictionary with the metrics names and values.
        The value can be either the metric itself or a list of values for each
        epochs
        :type metrics: dict[str, float], dict[str, list[float]]
        :param epoch: the epoch of the correspondent value, in case the value be
        :type epoch: int
        :return: the formatted message
        :rtype: str
        """
        message = ":\n"
        for k, v in metrics.items():
            k = get_formatted_name(k, self.output_map)
            if epoch is None:
                message += METRIC_LOG_PATTERN.format(v, k)
            else:
                message += METRIC_LOG_PATTERN.format(v[epoch], k)
        return message

    def on_epoch_end(self, epoch, logs=None):
        """
        Method called on the end of each epoch.

        :param epoch: the number of the epoch
        :type epoch: int
        :param logs: a dict of data from other callbacks
        :type logs: dict
        """
        logger.info("Epochs %d/%s%s", epoch + 1, self.number_of_epochs,
                    self.format_metrics_for_epoch(logs))


class AbstractNeuralLogCallback(Callback):
    """
    Defines an abstract NeuralLog callback.
    """

    def __init__(self, train_command):
        """
        Creates an abstract NeuralLog callback.

        :param train_command: the train command
        :type train_command: Train
        """
        super(AbstractNeuralLogCallback, self).__init__()
        self.train_command = train_command


@neural_log_callback("link_prediction_callback")
class LinkPredictionCallback(AbstractNeuralLogCallback):
    """
    Evaluates the model against the link predictions metrics: the mean
    reciprocal, the mean reciprocal rank and the hit at top k accuracy.

    This class appends the evaluation values into the logs dictionary, so other
    callbacks may use it.
    """

    # noinspection PyUnusedLocal
    def __init__(self, train_command, dataset, top_k=10, filtered=True,
                 period=1, rank_method="optimistic", suffix=None, **kwargs):
        """
        Evaluates the model against the link prediction metrics.

        :param train_command: the train command instance
        :type train_command: Train
        :param dataset: the name of the data set: train, valid or test
        :type dataset: str
        :param top_k: the top elements of the rank to measure the top k
        accuracy. The default is `10`
        :type top_k: int
        :param filtered: if `true`, the true examples from the previous sets (
        considering the order: train, valid and test) will not be considered in
        the count of the rank. The default value is `True`
        :type filtered: bool
        :param period: the interval (number of epochs) between the
        evaluation. The default value is `1`.
        :type period: int
        :param rank_method: the rank method, one of: optimistic, pessimistic or
        average. The optimistic considers only the entities with the score
        greater than the evaluated object, to create the rank; the pessimistic
        considers the entities with the score equals to or grater than the score
        of the evaluated object; and the average considers the entities with
        score greater than the evaluated objects plus half the entities with
        the score equals to the evaluated object
        :type rank_method: str
        :param suffix: the suffix of the metric to append to the history log.
        If `None` the `dataset_name` will be used instead.
        :type suffix: str or None
        """
        super(LinkPredictionCallback, self).__init__(train_command)
        self.dataset_name = dataset
        self.model = self.train_command.model  # type: NeuralLogNetwork
        self.program = self.model.program  # type: NeuralLogProgram
        self.top_k = top_k
        # if isinstance(filtered, str):
        #     filtered = filtered.lower()
        # self.filtered = filtered not in FALSE_VALUES
        self.filtered = filtered
        self.filter_datasets = self._get_filter_datasets()
        self.period = period
        self.rank_method = rank_method
        self.rank_function = get_rank_function(rank_method)
        self.epochs_since_last_save = 0
        self.output_indices = []  # type: List[int]
        self.output_predicates = []  # type: List[Tuple[Predicate, bool]]
        self.filtered_objects = []  # type: List[Dict[int, Set[int]]]
        self.dataset = self.train_command.get_dataset(self.dataset_name)
        if suffix is None:
            suffix = dataset
        self.mean_rank_name = MEAN_RANK_METRIC_FORMAT.format(suffix)
        self.mean_reciprocal_rank_name = \
            MEAN_RECIPROCAL_RANK_METRIC_FORMAT.format(suffix)
        self.top_k_rank_name = \
            TOP_K_RANK_METRIC_FORMAT.format(suffix)
        self._get_output_indices()

    def _get_filter_datasets(self):
        if not self.filtered:
            return []
        if self.dataset_name == TRAIN_SET_NAME:
            return [TRAIN_SET_NAME]
        elif self.dataset_name == VALIDATION_SET_NAME:
            return [TRAIN_SET_NAME, VALIDATION_SET_NAME]
        else:
            return [TRAIN_SET_NAME, VALIDATION_SET_NAME, TEST_SET_NAME]

    def _get_output_indices(self):
        index = 0
        self.output_indices = []
        self.output_predicates = []
        self.filtered_objects = []
        for predicate, inverted in self.model.predicates:
            if predicate.arity != 2:
                continue
            self.output_indices.append(index)
            self.output_predicates.append((predicate, inverted))
            filtered_entities = self._get_filtered_entities(predicate, inverted)
            self.filtered_objects.append(filtered_entities)
            index += 1

    def _get_filtered_entities(self, predicate, inverted):
        """
        Gets a dictionary with the set of filtered entities for the target
        entity, for the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :param inverted: if the predicate is inverted
        :type inverted: bool
        :return: the dictionary
        :rtype: dict[int, set[int]]
        """
        filtered_entities = dict()  # type: Dict[int, Set[int]]
        for dataset in self.filter_datasets:
            examples = self.program.examples.get(dataset, dict())
            examples = examples.get(predicate, dict())  # type: Dict[Any, Atom]
            for example in examples.values():
                sub_index = self.program.get_index_of_constant(
                    predicate, 1 if inverted else 0,
                    example.terms[1 if inverted else 0])
                obj_index = self.program.get_index_of_constant(
                    predicate, 0 if inverted else 1,
                    example.terms[0 if inverted else 1])
                filtered_entities.setdefault(sub_index, set()).add(obj_index)

        return filtered_entities

    def evaluate(self):
        """
        Evaluates the model on the link prediction metrics with the target
        dataset.

        For each binary predicate in the output of the model, it returns
        three values: (1) the `mean rank`, which is the mean of the rank of the
        true examples; (2) the `mean reciprocal rank`, which is the mean of the
        inverse of rank of the true examples; and (3) the `top @ k` metric,
        which is the percentage of the true examples whose rank is less than
        `k`.

        :return: three arrays, one for each metric
        :rtype: (np.ndarray, np.ndarray, np.ndarray)
        """
        if self.dataset is None:
            return
        number_of_indices = len(self.output_indices)
        rank = np.zeros(number_of_indices, dtype=np.int64)
        reciprocal_rank = np.zeros(number_of_indices, dtype=np.float64)
        top_hits = np.zeros(number_of_indices, dtype=np.int64)
        total_count = np.zeros(number_of_indices, dtype=np.int64)
        for features, labels in self.dataset:
            y_scores = self.model.predict(features)
            if len(self.model.predicates) == 1:
                y_scores = [y_scores]
                features = [features]
            for i in range(len(self.output_indices)):
                index = self.output_indices[i]
                for feature, y_true, y_score in \
                        zip(features[index], labels[index], y_scores[index]):
                    x = feature.numpy()
                    if x.max() == 0.0:
                        continue
                    y_true = y_true.numpy()
                    subject_index = np.argmax(x)
                    positive_objects = np.reshape(np.argwhere(y_true > 0.0), -1)
                    if len(positive_objects) == 0:
                        continue
                    # noinspection PyTypeChecker
                    filtered_objects = \
                        self.filtered_objects[i].get(subject_index, set())

                    for object_index in positive_objects:
                        bigger, equals = get_bigger_and_equals_counts(
                            y_score, object_index, filtered_objects)
                        object_rank = self.rank_function(bigger, equals)
                        rank[i] += object_rank
                        reciprocal_rank[i] += 1.0 / object_rank
                        if object_rank <= self.top_k:
                            top_hits[i] += 1.0
                        total_count[i] += 1

        mean_rank = rank / total_count
        mean_reciprocal_rank = reciprocal_rank / total_count
        top_hits_accuracy = top_hits / total_count

        return mean_rank, mean_reciprocal_rank, top_hits_accuracy, total_count

    def on_epoch_end(self, epoch, logs=None):
        """
        Method called on the end of each epoch.

        :param epoch: the number of the epoch
        :type epoch: int
        :param logs: a dict of data from other callbacks
        :type logs: dict
        """
        self.epochs_since_last_save += 1
        if logs is None:
            return

        if self.epochs_since_last_save >= self.period or \
                self.train_command.epochs == epoch + 1:
            self.epochs_since_last_save = 0
            metrics = self.evaluate()
            mean_rank = metrics[0]
            mean_reciprocal_rank = metrics[1]
            top_k_rank = metrics[2]
            total_count = metrics[3]
            for i in range(len(self.output_indices)):
                out = self.output_indices[i] + 1
                mean_key = LOG_FORMAT.format(out, self.mean_rank_name)
                mrr_key = LOG_FORMAT.format(out, self.mean_reciprocal_rank_name)
                top_key = LOG_FORMAT.format(out, self.top_k_rank_name)

                logs[mean_key] = mean_rank[i]
                logs[mrr_key] = mean_reciprocal_rank[i]
                logs[top_key] = top_k_rank[i]

            total = np.sum(total_count)
            weighted_mean_rank = np.sum(mean_rank * total_count) / total
            weighted_mrr = np.sum(mean_reciprocal_rank * total_count) / total
            weighted_top_k = np.sum(top_k_rank * total_count) / total
            logs[self.mean_rank_name] = weighted_mean_rank
            logs[self.mean_reciprocal_rank_name] = weighted_mrr
            logs[self.top_k_rank_name] = weighted_top_k


@neural_log_callback("roc_auc")
class AreaUnderCurveROC(AbstractNeuralLogCallback):
    """
    Evaluates the model against the Area Under the ROC Curve.

    This class appends the evaluation values into the logs dictionary, so other
    callbacks may use it.
    """

    # noinspection PyUnusedLocal
    def __init__(self, train_command, dataset,
                 positive_label=1.0, negative_label=0.0,
                 suffix=None, period=1, **kwargs):
        """
        Evaluates the model against the Area Under the ROC Curve.

        :param train_command: the train command instance
        :type train_command: Train
        :param dataset: the name of the data set: train, valid or test
        :type dataset: str
        :param suffix: the suffix of the metric to append to the history log.
        :param positive_label: the value of the positive label
        :type positive_label: float
        :param negative_label: the value of the negative label
        :type negative_label: float
        If `None` the `dataset_name` will be used instead.
        :type suffix: str or None
        :param period: the interval (number of epochs) between the
        evaluation. The default value is `1`.
        :type period: int
        """
        super(AreaUnderCurveROC, self).__init__(train_command)
        self.dataset_name = dataset
        if suffix is None:
            suffix = dataset
        self.metric_name = "roc_auc_{}".format(suffix)
        self.dataset = getattr(self.train_command, self.dataset_name)
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.period = period
        self.epochs_since_last_save = 0
        self.examples = self._build_examples()

    def _build_examples(self):
        model = self.train_command.model  # type: NeuralLogNetwork
        data_examples = model.program.examples[self.dataset_name]
        examples = dict()
        # neural_program.examples[set_name][predicate]
        for predicate, inverted in model.predicates:
            examples_by_predicate = dict()
            for example in data_examples[predicate].values():
                in_term = example.terms[-1 if inverted else 0]
                out_term = example.terms[0 if inverted else -1]
                weight = example.weight
                if weight == self.positive_label:
                    w = 1
                elif weight == self.negative_label:
                    w = 0
                else:
                    continue
                examples_by_predicate.setdefault(in_term, dict())[out_term] = w

            examples[(predicate, inverted)] = examples_by_predicate
        return examples

    def evaluate(self):
        """
        Evaluates the model on the Area Under the ROC Curve.

        For each predicate in the output of the model, it returns
        two values: (1) the label of the examples; (2) the score of the
        examples, which is the predictions of the model.

        :return: two arrays, one for scores and other for labels
        :rtype: dict[(Predicate, bool), (np.ndarray, np.ndarray)]
        """
        if self.dataset is None:
            return
        results = dict()
        model = self.train_command.model  # type: NeuralLogNetwork
        neural_program = model.program
        for predicate in model.predicates:
            results[predicate] = ([], [])

        for features, labels in self.dataset:
            y_scores = model.predict(features)
            if len(model.predicates) == 1:
                y_scores = [y_scores]
                features = [features]
            for i in range(len(model.predicates)):
                predicate = model.predicates[i]
                examples = self.examples[predicate]
                result = results[predicate]
                for feature, y_true, y_score in zip(
                        features[i], labels[i], y_scores[i]):
                    x = feature.numpy()
                    if x.max() == 0.0:
                        continue
                    subject_index = np.argmax(x)
                    # noinspection PyTypeChecker
                    # subject = neural_program.iterable_constants[subject_index]
                    subject = neural_program.get_constant_by_index(
                        predicate[0], 0, subject_index)
                    for obj, label in examples.get(subject, dict()).items():
                        obj_index = neural_program.get_index_of_constant(
                            predicate[0], 1, obj)
                        result[0].append(label)
                        result[1].append(y_score[obj_index])

        return results

    def on_epoch_end(self, epoch, logs=None):
        """
        Method called on the end of each epoch.

        :param epoch: the number of the epoch
        :type epoch: int
        :param logs: a dict of data from other callbacks
        :type logs: dict
        """
        self.epochs_since_last_save += 1
        if logs is None:
            return

        if self.epochs_since_last_save >= self.period or \
                self.train_command.epochs == epoch + 1:
            self.epochs_since_last_save = 0
            model = self.train_command.model  # type: NeuralLogNetwork
            metrics = self.evaluate()
            total = 0
            total_auc = 0.0
            for i in range(len(model.predicates)):
                predicate = model.predicates[i]
                metric_key = LOG_FORMAT.format(i + 1, self.metric_name)
                y_true, y_score = metrics[predicate]
                auc = roc_auc_score(y_true, y_score)
                logs[metric_key] = auc
                length = len(y_true)
                total_auc += auc * length
                total += length

            logs[self.metric_name] = total_auc / total
