"""
Command Line Interface command to train the model.
"""

import argparse
import logging
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from src.knowledge.program import NeuralLogProgram, print_neural_log_program, \
    DEFAULT_PARAMETERS
from src.network import trainer
from src.network.callbacks import get_formatted_name
from src.network.dataset import print_neural_log_predictions
from src.network.network import LossMaskWrapper
from src.network.network_functions import get_loss_function, CRFLogLikelihood
from src.network.trainer import Trainer
from src.run.command import Command, command, create_log_file, \
    TRAIN_SET_NAME, VALIDATION_SET_NAME, TEST_SET_NAME
from src.util import print_args
from src.util.file import read_logic_program_from_file

METRIC_FILE_PREFIX = "metric_"
LOGIC_PROGRAM_EXTENSION = ".pl"

TAB_SIZE = 4

COMMAND_NAME = "train"

logger = logging.getLogger(__name__)


def get_clauses(filepath):
    """
    Gets the clauses from the file in `filepath`.

    :param filepath: the filepath
    :type filepath: str
    :return: the clauses
    :rtype: List[Clause]
    """
    start_func = time.perf_counter()
    clauses = read_logic_program_from_file(filepath)
    end_func = time.perf_counter()

    logger.info("File:\t%s", filepath)
    logger.info("\t- Total reading time:\t%0.3fs",
                end_func - start_func)

    return clauses


def format_arguments(message, arguments):
    """
    Formats the arguments for the help message.

    :param message: the initial message
    :type message: str
    :param arguments: the arguments to be formatted
    :type arguments: list[list[str]] or list[tuple[str]]
    :return: the formatted message
    :rtype: str
    """
    formatted = message
    formatted += "\n\n"
    formatted += "The following parameters can be set in the logic file " \
                 "by using the special\npredicate set_parameter or " \
                 "set_predicate_parameter*.\n" \
                 "Syntax:\n\n" \
                 "set_parameter(<name>, <value>).\nor\n" \
                 "set_parameter(<name>, class_name, " \
                 "<class_name>).\n" \
                 "set_parameter(<name>, config, <config_1>, " \
                 "<value_1>).\n...\n" \
                 "set_parameter(<name>, config, <config_n>, " \
                 "<value_n>).\n\nor\n\n" \
                 "set_predicate_parameter(<predicate>, <name>, " \
                 "<value>).\nor\n" \
                 "set_predicate_parameter(<predicate>, <name>, class_name, " \
                 "<class_name>).\n" \
                 "set_predicate_parameter(<predicate>, <name>, config, " \
                 "<config_1>, " \
                 "<value_1>).\n...\n" \
                 "set_predicate_parameter(<predicate>, <name>, config, " \
                 "<config_n>, " \
                 "<value_n>).\n\n" \
                 "One can use $<predicate>[<index>] to access the size of " \
                 "the predicate term\nwhen setting parameters."
    formatted += "\n\n"
    max_key_size = max(map(lambda x: len(x[0]), arguments))
    stride = max_key_size + TAB_SIZE
    for argument in arguments:
        key, value = argument[0], argument[1]
        if len(argument) > 2 and argument[2]:
            key += "*"
        formatted += key + " " * (stride - len(key) - 1)
        length = 0
        for word in value.split(" "):
            length += len(word) + 1
            if length > 79 - stride:
                length = len(word) + 1
                formatted += "\n"
                formatted += " " * stride
            else:
                formatted += " "
            formatted += word
        formatted += "\n\n"
    formatted += "* this feature may be set individually for each " \
                 "predicate.\n" \
                 "If it is not defined for a specific predicate,\n" \
                 "the default globally defined value will be used."
    formatted += "\n\n"
    return formatted


def find_best_model(checkpoint, history):
    """
    Finds the best model saved by the checkpoint.

    :param checkpoint: the checkpoint
    :type checkpoint: ModelCheckpoint
    :param history: a dictionary with the metrics and their values for
    each epoch.
    :type history: dict[str, np.ndarray]
    :return: the path of the best model
    :rtype: str or None
    """
    if checkpoint.save_best_only:
        return checkpoint.filepath

    period = checkpoint.period
    monitor = checkpoint.monitor

    best = checkpoint.best
    monitor_op = checkpoint.monitor_op

    values = history.get(monitor, None)
    if values is None:
        return None
    best_epoch = 0
    for i in range(len(values)):
        if monitor_op(values[i], best):
            best = values[i]
            best_epoch = i

    return checkpoint.filepath.format(epoch=(best_epoch + 1) * period)


def deserialize_loss(loss_function):
    """
    Deserializes the loss functions.

    :param loss_function: the loss functions
    :type loss_function: str or dict
    :return: the deserialized loss functions
    :rtype: function or dict[function]
    """
    if isinstance(loss_function, dict):
        result = dict()
        for key, value in loss_function.items():
            result[key] = get_loss_function(value)
    else:
        result = get_loss_function(loss_function)
    return result


@command(COMMAND_NAME)
class Train(Command):
    """
    Trains the neural network.
    """

    def __init__(self, program, args, direct=False):
        super().__init__(program, args, direct)
        self.neural_program = NeuralLogProgram()
        self.train_set = None
        self.validation_set = None
        self.test_set = None

    # noinspection PyMissingOrEmptyDocstring,DuplicatedCode
    def build_parser(self):
        program = self.program
        if not self.direct:
            program += " {}".format(COMMAND_NAME)
        # noinspection PyTypeChecker
        parser = argparse.ArgumentParser(
            prog=program,
            description=self.get_command_description(),
            formatter_class=argparse.RawDescriptionHelpFormatter)

        # Input
        parser.add_argument('--program', '-p', metavar='program',
                            type=str, required=True, nargs="+",
                            help="The program file(s)")
        parser.add_argument('--train', '-t', metavar='train',
                            type=str, required=False, nargs="+", default=[],
                            help="The train file(s)")
        parser.add_argument('--validation', '-valid', metavar='validation',
                            type=str, required=False, nargs="+", default=[],
                            help="The validation file(s)")
        parser.add_argument('--test', '-test', metavar='test',
                            type=str, required=False, nargs="+", default=[],
                            help="The test file(s)")
        parser.add_argument('--loadModel', '-l', metavar='loadModel',
                            type=str, default=None, required=False,
                            help="If set, loads the model from the path and "
                                 "continues from the loaded model")

        # Output
        parser.add_argument("--outputPath", "-o", metavar='outputPath',
                            type=str, default=None, required=False,
                            help="The path to save the outputs")
        parser.add_argument("--lastModel", "-lm", metavar='lastModel',
                            type=str, default=None, required=False,
                            help="The path to save the last learned model. "
                                 "If `outputPath` is given, "
                                 "this path will be relative to it")
        parser.add_argument("--lastProgram", "-lp", metavar='lastProgram',
                            type=str, default=None, required=False,
                            help="The name of the file to save the last "
                                 "learned program. If `outputPath` is given, "
                                 "this path will be relative to it")
        parser.add_argument("--lastInference", "-li", metavar='lastInference',
                            type=str, default=None, required=False,
                            help="The prefix of the file to save the "
                                 "inferences of the last learned program. "
                                 "The name of the dataset and the `.pl` "
                                 "extension will be appended to it. "
                                 "If `outputPath` is given, this path will "
                                 "be relative to it")

        # Log
        parser.add_argument("--logFile", "-log", metavar='file',
                            type=str, default=None,
                            help="The file path to save the log into")
        parser.add_argument("--tensorBoard", "-tb", metavar='file',
                            type=str, default=None,
                            help="Creates a log event for the TensorBoard "
                                 "on the given path")
        parser.add_argument("--verbose", "-v", dest="verbose",
                            action="store_true",
                            help="Activated a verbose log")
        parser.set_defaults(verbose=False)
        return parser

    # noinspection PyMissingOrEmptyDocstring
    def get_command_description(self):
        message = super().get_command_description()
        arguments = list(
            map(lambda x: (x[0], x[2], x[3] if len(x) > 3 else True),
                DEFAULT_PARAMETERS))
        arguments += trainer.PARAMETERS
        return format_arguments(message, arguments)

    def _read_parameters(self):
        """
        Reads the default parameters found in the program
        """
        self.trainer.read_parameters()

        print_args(self.trainer.parameters, logger)

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def parse_args(self):
        super().parse_args()

        # Log
        args = self.parser.parse_args(self.args)
        log_file = args.logFile
        create_log_file(log_file)
        print_args(args, logger)
        self.tensor_board = args.tensorBoard

        # Input
        self.program_files = args.program
        self.train_files = args.train
        self.validation_files = args.validation
        self.test_files = args.test
        self.load_model = args.loadModel
        self.train = len(self.train_files) > 0
        self.valid = len(self.validation_files) > 0
        self.test = len(self.test_files) > 0

        # Output
        self.output_path = args.outputPath
        self.last_model = args.lastModel
        self.last_program = args.lastProgram
        self.last_inference = args.lastInference
        self.verbose = args.verbose
        if self.verbose:
            logger.setLevel(logging.DEBUG)
            # src.run.H1.setLevel(logging.DEBUG)

    def build(self):
        """
        Builds the neural network and prepares for training.
        """
        self._read_clauses_from_file()
        self._build_model()

    def _read_clauses_from_file(self):
        """
        Read the clauses from the files.
        """
        logger.info("Reading input files...")
        start_func = time.perf_counter()
        self._read_input_file(self.program_files, "program")
        end_program = time.perf_counter()
        end_train = end_program
        if self.train:
            self._read_input_file(self.train_files, TRAIN_SET_NAME)
            end_train = time.perf_counter()
        end_validation = end_train
        end_test = end_train
        end_reading = end_train
        if self.valid > 0:
            self._read_input_file(self.validation_files, VALIDATION_SET_NAME)
            end_validation = time.perf_counter()
            end_reading = end_validation
        if self.test > 0:
            self._read_input_file(self.test_files, TEST_SET_NAME)
            end_test = time.perf_counter()
            end_reading = end_test
        self.neural_program.build_program()
        end_func = time.perf_counter()
        # logger.info("Total number of predictable constants:\t%d",
        #             len(self.neural_program.iterable_constants))
        logger.info("Program reading time:   \t%0.3fs",
                    end_program - start_func)
        if self.train:
            logger.info("Train reading time:     \t%0.3fs",
                        end_train - end_program)
        if self.valid:
            logger.info("Validation reading time:\t%0.3fs",
                        end_validation - end_train)
        if self.test:
            logger.info("Test reading time:      \t%0.3fs",
                        end_test - end_validation)
        logger.info("Building program time:  \t%0.3fs",
                    end_func - end_reading)
        logger.info("Total reading time:     \t%0.3fs",
                    end_reading - start_func)

    def _read_input_file(self, program_files, name):
        logger.info("Reading %s...", name)
        for file in program_files:
            file_clauses = get_clauses(file)
            self.neural_program.add_clauses(file_clauses, example_set=name)

    def _build_model(self):
        """
        Builds and compiles the model.
        """
        start_func = time.perf_counter()
        logger.info("Building model...")
        self.trainer = Trainer(self.neural_program, self.output_path)
        self.trainer.init_model()
        self.neural_dataset = self.trainer.build_dataset()
        self.model = self.trainer.model
        self.model.build_layers(self.neural_dataset.get_target_predicates())
        self._read_parameters()
        self.trainer.log_parameters(
            ["clip_labels", "loss_function", "optimizer",
             "regularizer" "metrics", "inverse_relations"],
            self.trainer.output_map.inverse
        )
        self.trainer.compile_module()

        if self.load_model is not None:
            self.model.load_weights(self.load_model)

        end_func = time.perf_counter()

        logger.info("\nModel building time:\t%0.3fs", end_func - start_func)

    def fit(self):
        """
        Trains the neural network.
        """
        start_func = time.perf_counter()
        logger.info("Training the model...")
        self.trainer.log_parameters(["epochs", "validation_period"])
        self.trainer.build_callbacks(
            train_command=self, tensor_board=self.tensor_board)
        self.trainer.log_parameters(["callback"])
        history = self.trainer.fit(self.train_set, self.validation_set)
        end_func = time.perf_counter()
        logger.info("Total training time:\t%0.3fs", end_func - start_func)

        return history

    def _build_examples_set(self):
        start_func = time.perf_counter()
        logger.info("Creating training dataset...")
        shuffle = self.trainer.parameters["shuffle"]
        batch_size = self.trainer.parameters["batch_size"]
        self.trainer.log_parameters(["dataset_class", "batch_size", "shuffle"])
        end_func = time.perf_counter()
        train_set_time = 0
        validation_set_time = 0
        test_set_time = 0

        if self.train:
            self.train_set = self.neural_dataset.get_dataset(
                example_set=TRAIN_SET_NAME,
                batch_size=batch_size,
                shuffle=shuffle)
            end_train = time.perf_counter()
            train_set_time = end_train - start_func
            end_func = end_train
        if self.valid:
            self.validation_set = self.neural_dataset.get_dataset(
                example_set=VALIDATION_SET_NAME, batch_size=batch_size)
            end_valid = time.perf_counter()
            validation_set_time = end_valid - end_func
            end_func = end_valid
        if self.test:
            self.test_set = self.neural_dataset.get_dataset(
                example_set=TEST_SET_NAME, batch_size=batch_size)
            end_test = time.perf_counter()
            test_set_time = end_test - end_func
            end_func = end_test

        if self.train:
            logger.info("Train dataset creating time:      \t%0.3fs",
                        train_set_time)
        if self.valid:
            logger.info("Validation dataset creation time: \t%0.3fs",
                        validation_set_time)
        if self.test:
            logger.info("Test dataset creation time:       \t%0.3fs",
                        test_set_time)

        logger.info("Total dataset creation time:      \t%0.3fs",
                    end_func - start_func)

    # noinspection PyMissingOrEmptyDocstring
    def run(self):
        self.build()
        history = None
        self._build_examples_set()
        self._save_transitions("transition_before.txt")
        if self.train:
            history = self.fit()
            if logger.isEnabledFor(logging.INFO):
                hist = history.history
                hist = dict(map(
                    lambda x: (get_formatted_name(
                        x[0], self.trainer.output_map.inverse), x[1]),
                    hist.items()))
                logger.info("\nHistory:")
                for key, value in hist.items():
                    logger.info("%s: %s", key, value)
                logger.info("")

        logger.info("Saving data...")
        start_save = time.perf_counter()
        if history is not None and self.last_model is not None:
            filepath = self._get_output_path(self.last_model)
            self.model.save_weights(filepath)
            logger.info("\tLast model saved at:\t{}".format(filepath))
            for metric in history.history:
                array = np.array(history.history[metric])
                # noinspection PyTypeChecker
                metric = get_formatted_name(
                    metric, self.trainer.output_map.inverse)
                metric = METRIC_FILE_PREFIX + metric
                metric_path = os.path.join(self.output_path,
                                           "{}.txt".format(metric))
                # noinspection PyTypeChecker
                np.savetxt(metric_path, array, fmt="%0.8f")

        if not self.train and self.load_model is None:
            return

        self.save_program(self.last_program)
        self.save_inferences(self.last_inference)

        if history is not None:
            logger.info("")
            for key, value in self.trainer.best_models.items():
                path = find_best_model(value, history.history)
                if path is None:
                    continue
                self.model.load_weights(path)
                self.save_program(key + "program.pl")
                if self.last_inference is not None:
                    self.save_inferences(key)
                logger.info("\tBest model for {} saved at:\t{}".format(
                    key, path))
        end_save = time.perf_counter()
        self._save_transitions("transition_after.txt")

        logger.info("Total data saving time:\t%0.3fs", end_save - start_save)

    def _save_transitions(self, filename):
        """
        Saves the transitions to file.
        """
        loss_function = self.trainer.parameters["loss_function"]
        if isinstance(loss_function, LossMaskWrapper):
            loss_function = loss_function.function
        if isinstance(loss_function, CRFLogLikelihood):
            filepath = self._get_output_path(filename)
            transition = loss_function.transition_params.numpy()
            np.savetxt(filepath, transition)
            logger.info("transitions:\n%s", transition)

    def save_program(self, program_path):
        """
        Saves the program of the current model.

        :param program_path: the path to save the program
        :type program_path: str
        """
        if program_path is not None:
            self.model.update_program()
            output_program = self._get_output_path(program_path)
            output_file = open(output_program, "w")
            print_neural_log_program(self.neural_program, output_file)
            output_file.close()
            logger.info("\tProgram saved at:\t{}".format(output_program))

    def save_inferences(self, file_prefix):
        """
        Saves the inferences of the current model of the different datasets.

        :param file_prefix: the prefix of the path to be appended with
        the dataset's name
        :type file_prefix: str
        """
        if file_prefix is not None:
            if self.train or self.valid or self.test:
                logger.info("\tInferences saved at:")

            if self.train:
                self._save_inference_for_dataset(file_prefix, TRAIN_SET_NAME)

            if self.valid:
                self._save_inference_for_dataset(
                    file_prefix, VALIDATION_SET_NAME)

            if self.test:
                self._save_inference_for_dataset(file_prefix, TEST_SET_NAME)

    def _save_inference_for_dataset(self, file_prefix, dataset_name):
        tab = "\t\t"
        if dataset_name == VALIDATION_SET_NAME:
            tab = "\t"
        output = self._get_inference_filename(file_prefix, dataset_name)
        logger.info("\t\t{}:{}{}".format(dataset_name, tab, output))
        self.write_neural_log_predictions(output, dataset_name)

    def _get_inference_filename(self, prefix, dataset):
        return self._get_output_path(prefix + dataset + LOGIC_PROGRAM_EXTENSION)

    def _get_output_path(self, suffix):
        if self.output_path is not None:
            return os.path.join(self.output_path, suffix)
        return suffix

    def get_dataset(self, name):
        """
        Gets the dataset based on the name.

        :param name: the name of the dataset
        :type name: str
        :return: the dataset
        :rtype: tf.data.Dataset or None
        """
        if name == TRAIN_SET_NAME:
            return self.train_set
        if name == VALIDATION_SET_NAME:
            return self.validation_set
        if name == TEST_SET_NAME:
            return self.test_set
        return None

    def write_neural_log_predictions(self, filepath, dataset_name):
        """
        Writes the predictions of the model, for the dataset to the `filepath`.

        :param filepath: the file path
        :type filepath: str
        :param dataset_name: the name of the dataset
        :type dataset_name: str
        """
        dataset = self.get_dataset(dataset_name)
        writer = open(filepath, "w")
        print_neural_log_predictions(self.model, self.neural_program,
                                     self.neural_dataset, dataset, writer,
                                     dataset_name)
        writer.close()
