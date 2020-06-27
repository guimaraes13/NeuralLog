"""
Handles the training of a NeuralLog network.
"""
import logging
import os
from functools import reduce
from typing import Optional, Tuple, Dict

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import TensorBoard

from src.knowledge.program import get_predicate_from_string, BiDict, \
    NeuralLogProgram
from src.language.language import Predicate
from src.network.callbacks import EpochLogger, get_neural_log_callback, \
    AbstractNeuralLogCallback
from src.network.dataset import get_dataset_class, NeuralLogDataset
from src.network.network import LossMaskWrapper, NeuralLogNetwork
from src.network.network_functions import get_loss_function
from src.util import print_args

DEFAULT_LOSS = "mean_squared_error"
DEFAULT_OPTIMIZER = "sgd"
DEFAULT_REGULARIZER = None
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUMBER_OF_EPOCHS = 10
DEFAULT_VALID_PERIOD = 1
DEFAULT_INVERTED_RELATIONS = True
DEFAULT_MASK_PREDICTIONS = False
DEFAULT_CLIP_LABELS = False
DEFAULT_SHUFFLE = False

logger = logging.getLogger(__name__)

PARAMETERS = [
    ("inverse_relations", "if `True`, creates also the inverted relation for "
                          "each output predicate. The default value is: "
                          "{}".format(DEFAULT_INVERTED_RELATIONS)),
    ("loss_function", "the loss function of the neural network and, possibly, "
                      "its options. The default value is: {}. It can be "
                      "individually specified for each predicate, just put "
                      "another term with the name of the predicate"
                      "".format(DEFAULT_LOSS.replace("_", " "))),
    ("metrics", "the metric functions to eval the neural network and, "
                "possibly, its options. The default value is the loss"
                "function, which is always appended to the metrics. It can "
                "be individually specified for each predicate, just put "
                "another term with the name of the predicate"),
    ("optimizer", "the optimizer for the training and, possibly, its options. "
                  "The default value is: {}".format(DEFAULT_OPTIMIZER)),
    ("regularizer", "specifies the regularizer, it can be `l1`, `l2` or "
                    "`l1_l2`. The default value is: "
                    "{}".format(DEFAULT_REGULARIZER)),
    ("batch_size", "the batch size. The default value is: "
                   "{}".format(DEFAULT_BATCH_SIZE)),
    ("epochs", "the number of epochs. The default value is: "
               "{}".format(DEFAULT_NUMBER_OF_EPOCHS)),
    ("shuffle", "if set, shuffles the examples of the dataset for each "
                "iteration. This option is computationally expensive"),
    ("validation_period", "the interval (number of epochs) between the "
                          "validation. The default value is:"
                          "{}".format(DEFAULT_VALID_PERIOD)),
    ("callback", "a dictionary of callbacks to be used on training. "
                 "The default value is `None`"),
    ("best_model", "a dictionary with keys matching pointing to "
                   "`ModelCheckpoints` in the callback dictionary. For each "
                   "entry, it will save the program and inference files ("
                   "with the value of the entry as prefix) based on the best "
                   "model saved by the checkpoint defined by the key. "
                   "The default value is `None`"),
    ("mask_predictions", "if `True`, it masks the output of the network, "
                         "during the training phase. Before the loss "
                         "function, it sets the predictions of unknown "
                         "examples to `0` by multiplying the output of the "
                         "network by the square of the labels. In order to "
                         "this method work, the labels must be: `1`, "
                         "for positive examples; `-1`, for negative examples; "
                         "and `0`, for unknown examples"),
    ("clip_labels", "if `True`, clips the values of the labels to [0, 1]. "
                    "This is useful when one wants to keep the output of the "
                    "network in [0, 1], and also use the mask_predictions "
                    "features."),
]


def unique(elements):
    """
    Returns a list of unique elements from the `elements`.

    :param elements: the input elements.
    :type elements: list
    :return: the list of unique elements
    :rtype: list
    """
    element_set = set()
    unique_list = []
    for element in elements:
        if isinstance(element, dict):
            key = frozenset(element)
        else:
            key = element
        if key in element_set:
            continue
        unique_list.append(element)
        element_set.add(key)
    return unique_list


class Trainer:
    """
    Class to handle the training of a NeuralLog Network.
    """

    def __init__(self, neural_program, output_path):
        self.neural_program: NeuralLogProgram = neural_program
        self.output_path = output_path
        self.model: Optional[NeuralLogNetwork] = None
        self.parameters: Optional[Dict] = None
        self.output_map: BiDict[Tuple[Predicate, bool], str] = BiDict()
        "The map of the outputs of the neural network by the predicate"

        self.callbacks = []
        self.best_models = dict()
        self._neural_dataset: Optional[NeuralLogDataset] = None
        self._is_parameter_up_to_date = False

    # noinspection PyMissingOrEmptyDocstring
    @property
    def neural_dataset(self):
        self.build_dataset()
        return self._neural_dataset

    @neural_dataset.setter
    def neural_dataset(self, value):
        self._neural_dataset = value

    def init_model(self):
        """
        Initializes the NeuralLog model.
        """
        self.model = NeuralLogNetwork(self.neural_program, train=True,
                                      regularizer=self.regularizer)

    def compile_module(self):
        """
        Compiles the NeuralLog model.
        """
        if self.parameters is None:
            self.read_parameters()
        self.model.compile(
            loss=self.parameters["loss_function"],
            optimizer=self.parameters["optimizer"],
            metrics=self.parameters["metrics"]
        )

    def _get_output_map(self):
        output_map = BiDict()  # type: BiDict[Predicate, str]
        count = 1
        for predicate in self.model.predicates:
            output_map[predicate] = "output_{}".format(count)
            count += 1
        return output_map

    # noinspection PyMissingOrEmptyDocstring
    @property
    def regularizer(self):
        return self.neural_program.parameters.get(
            "regularizer", DEFAULT_REGULARIZER)

    def read_parameters(self):
        """
        Reads the default parameters found in the program
        """
        if self._is_parameter_up_to_date and self.neural_program.is_up_to_date:
            return
        self.output_map = self._get_output_map()
        self.parameters = dict(self.neural_program.parameters)
        self.parameters.setdefault(
            "inverse_relations", DEFAULT_INVERTED_RELATIONS)
        self.parameters.setdefault("mask_predictions", DEFAULT_MASK_PREDICTIONS)
        self.parameters.setdefault("shuffle", DEFAULT_SHUFFLE)
        self.parameters["loss_function"] = self._get_loss_function()
        self.parameters.setdefault("clip_labels", DEFAULT_CLIP_LABELS)
        self._wrap_mask_loss_functions()
        self.parameters["metrics"] = self._get_metrics()
        self.parameters.setdefault("optimizer", DEFAULT_OPTIMIZER)
        self.parameters.setdefault("regularizer", DEFAULT_REGULARIZER)
        self.parameters.setdefault("batch_size", DEFAULT_BATCH_SIZE)
        self.parameters.setdefault("epochs", DEFAULT_NUMBER_OF_EPOCHS)
        self.parameters.setdefault("validation_period", DEFAULT_VALID_PERIOD)
        self._is_parameter_up_to_date = True

    def _get_loss_function(self):
        """
        Gets the loss function.

        :return: the loss function for each output
        :rtype: str or dict[str, str]
        """
        loss_function = self.parameters.get("loss_function", DEFAULT_LOSS)
        if isinstance(loss_function, dict) and \
                "class_name" not in loss_function and \
                "config" not in loss_function:
            default_loss = DEFAULT_LOSS
            results = dict()
            for key, value in loss_function.items():
                key = get_predicate_from_string(key)
                has_not_match = True
                for predicate, output in self.output_map.items():
                    if key.equivalent(predicate[0]):
                        results.setdefault(output, value)
                        has_not_match = False
                if has_not_match:
                    default_loss = value
            for key in self.output_map.values():
                results.setdefault(key, default_loss)
            for key, value in results.items():
                results[key] = get_loss_function(value)
        else:
            results = get_loss_function(loss_function)
        return results

    def _wrap_mask_loss_functions(self):
        """
        Wraps the loss functions to mask the values of unknown examples.

        It multiplies the output of the network by the square of the labels. In
        order to this method work, the labels must be: `1`, for positive
        examples; `-1`, for negative examples; and `0`, for unknown examples.

        In this way, the square of the labels will be `1` for the positive and
        negative examples; and `0`, for the unknown examples. When multiplied by
        the prediction, the predictions of the unknown examples will be zero,
        thus, having no error and no gradient for those examples. While the
        predictions of the known examples will remain the same.
        """
        if not self.parameters["mask_predictions"]:
            return
        loss_function = self.parameters["loss_function"]
        label_function = None
        if self.parameters["clip_labels"]:
            label_function = lambda x: tf.clip_by_value(x, clip_value_min=0.0,
                                                        clip_value_max=1.0)
        if isinstance(loss_function, dict):
            functions = dict()
            for key, value in loss_function.items():
                functions[key] = LossMaskWrapper(value, label_function)
        else:
            # noinspection PyTypeChecker
            functions = LossMaskWrapper(loss_function, label_function)
        self.parameters["loss_function"] = functions

    def _get_metrics(self):
        """
        Gets the metrics.

        :rtype: str or dict[str, str]
        """
        metrics = self.parameters.get("metrics", None)
        loss = self.parameters["loss_function"]
        if isinstance(metrics, dict):
            results = dict()
            all_metrics = []
            for key, values in metrics.items():
                if isinstance(values, dict):
                    values = \
                        sorted(values.items(), key=lambda x: x[0])
                    values = list(map(lambda x: x[1], values))
                else:
                    values = [values]
                key = get_predicate_from_string(key)
                has_not_match = True
                for predicate, output in self.output_map.items():
                    if key.equivalent(predicate[0]):
                        metric = results.get(output, [])
                        results[output] = metric + values
                        has_not_match = False
                if has_not_match:
                    all_metrics.append((key, values))
            all_metrics = sorted(all_metrics, key=lambda x: x[0])
            all_metrics = list(map(lambda x: x[1], all_metrics))
            if len(all_metrics) > 0:
                all_metrics = reduce(list.__add__, all_metrics)
            for key in self.output_map.values():
                values = results.get(key, [])
                default_loss = loss.get(key) if isinstance(loss, dict) else loss
                results[key] = unique([default_loss] + all_metrics + values)
            return results
        elif metrics is None:
            if isinstance(loss, dict):
                return loss
            else:
                return [loss]
        else:
            if isinstance(loss, dict):
                results = dict()
                for key in loss.keys():
                    results[key] = [loss[key], self.parameters["metrics"]]
                return results
            else:
                return [loss, self.parameters["metrics"]]

    def log_parameters(self, parameter_keys, map_dict=None):
        """
        Logs the parameters.

        :param parameter_keys: the keys of the parameters to log.
        :type parameter_keys: collections.Collection[str]
        :param map_dict: a map of strings to replace the names of the outputs
        :type map_dict: dict[str, str]
        """
        if logger.isEnabledFor(logging.INFO):
            parameters = dict(filter(lambda x: x[0] in parameter_keys,
                                     self.parameters.items()))
            if len(parameters) == 0:
                return
            if map_dict is not None:
                # noinspection PyUnresolvedReferences
                for key, value in parameters.items():
                    if isinstance(value, dict):
                        new_value = dict()
                        for k, v in value.items():
                            k = map_dict.get(k, k)
                            if isinstance(k, tuple):
                                k = k[0].__str__() + (" (inv)" if k[1] else "")
                            new_value[k] = v
                        parameters[key] = new_value
            print_args(parameters, logger)

    def fit(self, train_set, validation_set=None):
        """
        Fits the model.

        :param train_set: the train set
        :type train_set: tf.data.Dataset
        :param validation_set: the validation set
        :type validation_set: tf.data.Dataset
        """
        return self.model.fit(
            train_set,
            epochs=self.parameters["epochs"],
            validation_data=validation_set,
            validation_freq=self.parameters["validation_period"],
            callbacks=self.callbacks
        )

    def build_callbacks(self, train_command=None, tensor_board=None):
        """
        Builds the callbacks.

        :param train_command: the train command class, if any
        :type train_command: Any
        :param tensor_board: Creates a log event for the TensorBoard
        on the given path, if not `None`
        :type tensor_board: Optional[str]
        """
        self.callbacks = []
        if tensor_board is not None:
            self.callbacks.append(TensorBoard(tensor_board))

        self._build_parameter_callbacks(train_command)

        self.callbacks.append(
            EpochLogger(self.parameters["epochs"], self.output_map.inverse))

    def _build_parameter_callbacks(self, train_command=None):
        callbacks_parameters = self.parameters.get("callback", None)
        if callbacks_parameters is None:
            return

        best_model_parameters = self.parameters.get("best_model", dict())
        for name, identifier in callbacks_parameters.items():
            if isinstance(identifier, dict):
                class_name = identifier["class_name"]
                config = identifier.get("config", dict())
            else:
                class_name = identifier
                config = dict()
            callback_class = get_neural_log_callback(class_name)
            if callback_class is None:
                continue
            config = self._adjust_config_for_callback(
                config, callback_class, train_command)
            callback = callback_class(**config)
            if isinstance(callback, ModelCheckpoint):
                best_model_name = best_model_parameters.get(name, None)
                if best_model_name is not None:
                    self.best_models[best_model_name] = callback
            self.callbacks.append(callback)

    def _adjust_config_for_callback(self, config, callback_class,
                                    train_command=None):
        config.setdefault("period", self.parameters.get("validation_period"))
        if issubclass(callback_class, AbstractNeuralLogCallback):
            config["train_command"] = train_command
        elif issubclass(callback_class, ModelCheckpoint):
            config.setdefault("save_best_only", True)
            config.setdefault("save_weights_only", True)
            has_no_filepath = "filepath" not in config
            config.setdefault("filepath", config["monitor"])
            config["filepath"] = self._get_output_path(config["filepath"])
            if not config["save_best_only"]:
                config["filepath"] = config["filepath"] + "_{epoch}"
            elif has_no_filepath:
                config["filepath"] = config["filepath"] + "_best"
            if "mode" not in config:
                if config["monitor"].startswith("mean_rank"):
                    config["mode"] = "min"
                else:
                    config["mode"] = "max"
        return config

    def _get_output_path(self, suffix):
        if self.output_path is not None:
            return os.path.join(self.output_path, suffix)
        return suffix

    def build_dataset(self):
        """
        Builds the NeuralLog dataset.

        :return: the NeuralLog dataset
        :rtype: NeuralLogDataset
        """
        if self._neural_dataset is not None:
            return self._neural_dataset
        inverse_relations = self.parameters.get(
            "inverse_relations", DEFAULT_INVERTED_RELATIONS)
        dataset_class = self.neural_program.parameters["dataset_class"]
        config = dict()
        if isinstance(dataset_class, dict):
            class_name = dataset_class["class_name"]
            config.update(dataset_class["config"])
        else:
            class_name = dataset_class
        config["program"] = self.neural_program
        config["inverse_relations"] = inverse_relations
        self._neural_dataset = get_dataset_class(class_name)(**config)
        return self._neural_dataset

    def get_dataset(self, name):
        """
        Gets the dataset by name.

        :param name: the name of the dataset
        :type name: str
        :return: the dataset
        :rtype: tf.data.Dataset
        """
        return self.neural_dataset.get_dataset(
            name, self.parameters["batch_size"], self.parameters["shuffle"]
        )
