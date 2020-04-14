"""
Command Line Interface command to output the model's prediction into NLP
format in order to be evaluated by the CoNLL evaluation script.
"""

import argparse
import fileinput
import logging
import re
import time

import numpy as np

from src.knowledge.program import NeuralLogProgram
from src.language.language import AtomClause, Atom, \
    Predicate
from src.network.dataset import get_dataset_class, WordCharDataset, log_viterbi
from src.network.network import NeuralLogNetwork
from src.run.command import Command, command, print_args, TEST_SET_NAME
from src.run.command.train import get_clauses, DEFAULT_BATCH_SIZE

SPLIT_VALUE = " "

DEFAULT_TAG = "O"

COMMAND_NAME = "output_nlp"

logger = logging.getLogger()

EXAMPLE_PREDICATE = "mega_example"
NUMBER = re.compile("[0-9]+")

DELIMITER = " "


def scape_word(word):
    """
    Scape the word
    """
    return word.replace("\\", "\\\\").replace("\"", "\\\"")


# noinspection DuplicatedCode
def get_examples(file_paths, target_predicate, split_value=SPLIT_VALUE):
    """
    Gets the examples from the input file and converts it to logic.

    :param split_value: the value to split the input
    :type split_value: str
    :param file_paths: the file paths
    :type file_paths: list[str]
    :param target_predicate: the name of the logic predicate
    :type target_predicate: str
    :return: the logic examples
    :rtype: list[AtomClause]
    """
    sentence_number = 0
    examples = []
    for line in fileinput.input(files=file_paths):
        line = line.strip()
        if line == "":
            sentence_number += 1
            continue
        fields = line.split(split_value)
        word = fields[0]
        word = scape_word(word)
        if NUMBER.search(word) is not None:
            word = "0"
        word = "\"" + word + "\""
        tag = "\"" + DEFAULT_TAG + "\""
        examples.append(AtomClause(Atom(
            EXAMPLE_PREDICATE,
            sentence_number,
            target_predicate,
            word,
            tag
        )))

    return examples


# noinspection DuplicatedCode
@command(COMMAND_NAME)
class OutputNLP(Command):
    """
    Trains the neural network.
    """

    def __init__(self, program, args, direct=False):
        super().__init__(program, args, direct)
        self.neural_program = NeuralLogProgram()
        self.parameters = None
        self.test_set = None

    # noinspection PyMissingOrEmptyDocstring
    def build_parser(self) -> argparse.ArgumentParser:
        program = self.program
        if not self.direct:
            program += " {}".format(COMMAND_NAME)
        parser = argparse.ArgumentParser(
            prog=program,
            description=self.get_command_description(),
            formatter_class=argparse.RawDescriptionHelpFormatter)

        # Input
        parser.add_argument('--program', '-p', metavar='program',
                            type=str, required=True, nargs="+",
                            help="The program file(s)")
        parser.add_argument('--targetPredicate', '-target', metavar='target',
                            type=str, required=True,
                            help="The target predicate")
        parser.add_argument('--test', '-t', metavar='test',
                            type=str, required=True, nargs="+",
                            help="The test file(s)")
        parser.add_argument('--loadModel', '-l', metavar='loadModel',
                            type=str, required=True,
                            help="If set, loads the model from the path and "
                                 "continues from the loaded model")

        parser.add_argument('--tabSeparator', '-tab', dest='tabSeparator',
                            action="store_true",
                            help="If set, use tab to separate the values of "
                                 "the input files")
        parser.set_defaults(tabSeparator=False)

        # Viterbi
        parser.add_argument('--transitionPredicate', '-transition',
                            metavar='transition', type=str,
                            required=False, default=None,
                            help="The name of the predicate that hold the "
                                 "transition matrix of the tags, in order to "
                                 "compute the Viterbi algorithm.")

        parser.add_argument('--initialPredicate', '-initial',
                            metavar='initial', type=str,
                            required=False, default=None,
                            help="The name of the predicate that hold the "
                                 "initial probabilities of the tags, in order "
                                 "to compute the Viterbi algorithm.")

        # Output
        parser.add_argument("--outputFile", "-o", metavar='outputFile',
                            type=str, required=True,
                            help="The path to save the outputs")

        # Log
        return parser

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def parse_args(self):
        # Log
        args = self.parser.parse_args(self.args)
        print_args(args)

        # Input
        self.program_files = args.program
        self.test_files = args.test
        self.load_model = args.loadModel
        self.test = len(self.test_files) > 0
        self.target_predicate = args.targetPredicate
        self.transition_predicate = None
        self.initial_predicate = None
        self.split_value = "\t" if args.tabSeparator else SPLIT_VALUE
        if args.transitionPredicate is not None:
            self.transition_predicate = Predicate(args.transitionPredicate, 2)
            if args.initialPredicate is not None:
                self.initial_predicate = Predicate(args.initialPredicate, 1)

        self.output_file = args.outputFile

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
        end_test = end_program
        end_reading = end_program
        if self.test > 0:
            logger.info("Reading %s...", TEST_SET_NAME)
            file_clauses = get_examples(self.test_files, self.target_predicate,
                                        self.split_value)
            # file_clauses = get_clauses(file)
            self.neural_program.add_clauses(
                file_clauses, example_set=TEST_SET_NAME)
            end_test = time.perf_counter()
            end_reading = end_test
        self.neural_program.build_program()
        end_func = time.perf_counter()
        # logger.info("Total number of predictable constants:\t%d",
        #             len(self.neural_program.iterable_constants))
        logger.info("Program reading time:   \t%0.3fs",
                    end_program - start_func)
        if self.test:
            logger.info("Test reading time:      \t%0.3fs",
                        end_test - end_program)
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
        self._create_dataset()
        self.model = NeuralLogNetwork(self.neural_dataset, train=True)
        self.model.build_layers()

        if self.load_model is not None:
            self.model.load_weights(self.load_model)

        end_func = time.perf_counter()

        logger.info("\nModel building time:\t%0.3fs", end_func - start_func)

    def _create_dataset(self):
        dataset_class = self.neural_program.parameters["dataset_class"]
        config = dict()
        if isinstance(dataset_class, dict):
            class_name = dataset_class["class_name"]
            config.update(dataset_class["config"])
        else:
            class_name = dataset_class
        config["program"] = self.neural_program
        config["inverse_relations"] = False
        self.neural_dataset = get_dataset_class(class_name)(**config)

    def _build_examples_set(self):
        start_func = time.perf_counter()
        logger.info("Creating training dataset...")
        batch_size = self.neural_program.parameters.get(
            "batch_size", DEFAULT_BATCH_SIZE)
        end_func = time.perf_counter()
        test_set_time = 0

        if self.test:
            self.test_set = self.neural_dataset.get_dataset(
                example_set=TEST_SET_NAME, batch_size=batch_size)
            end_test = time.perf_counter()
            test_set_time = end_test - end_func
            end_func = end_test

        if self.test:
            logger.info("Test dataset creation time:       \t%0.3fs",
                        test_set_time)

        logger.info("Total dataset creation time:      \t%0.3fs",
                    end_func - start_func)

    # noinspection PyMissingOrEmptyDocstring
    def run(self):
        self.build()
        self._build_examples_set()
        logger.info("Saving data...")
        start_save = time.perf_counter()

        logger.info("\tInferences saved at:")
        self._save_inference_for_dataset()

        end_save = time.perf_counter()
        logger.info("Total data saving time:\t%0.3fs", end_save - start_save)

    def _save_inference_for_dataset(self):
        logger.info("\t\t{}:\t\t{}".format(TEST_SET_NAME, self.output_file))
        input_file_it = iter(fileinput.input(files=self.test_files))
        writer = open(self.output_file, "w")
        neural_dataset = self.model.dataset  # type: WordCharDataset
        transition_matrix = None
        initial_probabilities = None
        if self.transition_predicate is not None:
            transition_matrix = self.neural_program.get_matrix_representation(
                self.transition_predicate).toarray()
            transition_matrix = np.log(transition_matrix)
            if self.initial_predicate is not None:
                initial_probabilities = \
                    self.neural_program.get_matrix_representation(
                        self.initial_predicate).toarray().squeeze()
                initial_probabilities = np.log(initial_probabilities)
        for features, _ in self.test_set:
            # For each sentence
            y_scores = self.model.predict(features)
            if len(self.model.predicates) == 1:
                y_scores = [y_scores]
                if self.model.predicates[0][0].arity < 3:
                    features = [features]
            for i in range(len(self.model.predicates)):
                # For each predicate
                predicate, inverted = self.model.predicates[i]
                predicate = Predicate(predicate.name, predicate.arity - 1)
                row_scores = y_scores[i]
                if len(row_scores.shape) == 3:
                    row_scores = np.squeeze(row_scores, axis=1)
                if transition_matrix is not None:
                    row_scores = log_viterbi(
                        transition_matrix,
                        np.log(row_scores), initial_probabilities)
                for j in range(len(row_scores)):
                    # For each word
                    y_score = row_scores[j]
                    offset = sum(self.model.input_sizes[:i])

                    last_feature = features[offset - 1][j].numpy()
                    pred_word = ""
                    for k in last_feature:
                        if k == neural_dataset.empty_char_index:
                            break
                        pred_word += \
                            self.neural_program.get_constant_by_index(
                                neural_dataset.character_predicate,
                                neural_dataset.character_predicate_index,
                                k
                            ).value

                    if transition_matrix is None:
                        pred_tag = self.neural_program.get_constant_by_index(
                            predicate, -1, y_score.argmax()).value
                    else:
                        pred_tag = self.neural_program.get_constant_by_index(
                            predicate, -1, y_score).value

                    word, tag = next(input_file_it).strip().split(
                        self.split_value)
                    print(word, tag, pred_tag, sep=DELIMITER, file=writer)
            print("", file=writer)
            try:
                next(input_file_it)
            except StopIteration:
                break

        writer.close()
