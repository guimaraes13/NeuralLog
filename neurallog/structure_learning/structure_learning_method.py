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
The methods to call the structure learning.
"""
import copy
import logging
import os
import re
from abc import ABC, abstractmethod
from itertools import chain
from typing import List, Optional, Iterable

import yaml

from neurallog.knowledge.examples import Examples, ExampleIterator, \
    LimitedIterator, \
    ExamplesInferences
from neurallog.knowledge.manager.example_manager import ReviseAllIncomingExample
from neurallog.knowledge.program import NeuralLogProgram
from neurallog.knowledge.theory.evaluation.metric.theory_metric import \
    TheoryMetric, RocCurveMetric, PrecisionRecallCurveMetric, \
    LogLikelihoodMetric
from neurallog.knowledge.theory.evaluation.theory_evaluator import \
    TheoryEvaluator
from neurallog.knowledge.theory.manager.revision.operator.revision_operator \
    import BottomClauseBoundedRule
from neurallog.knowledge.theory.manager.revision.revision_manager import \
    RevisionManager
from neurallog.knowledge.theory.manager.revision.revision_operator_evaluator \
    import RevisionOperatorEvaluator
from neurallog.knowledge.theory.manager.revision.revision_operator_selector \
    import SelectFirstRevisionOperator
from neurallog.knowledge.theory.manager.revision.sample_selector import \
    AllRelevantSampleSelector
from neurallog.knowledge.theory.manager.theory_revision_manager import \
    TheoryRevisionManager
from neurallog.language.language import Clause, AtomClause
from neurallog.run.command import TRAIN_SET_NAME, VALIDATION_SET_NAME
from neurallog.structure_learning.engine_system_translator import \
    EngineSystemTranslator
from neurallog.structure_learning.structure_learning_system import \
    StructureLearningSystem
from neurallog.util import Initializable
from neurallog.util.file import read_logic_program_from_file, \
    print_predictions_to_file, print_tsv_file_file
from neurallog.util.statistics import RunStatistics, IterationStatistics
from neurallog.util.time_measure import TimeMeasure, performance_time, \
    IterationTimeStampFactory, IterationTimeMessage, RunTimestamps

STATISTICS_FILE_NAME = "statistics.yaml"

TRAIN_INFERENCE_FILE = "train.inference.pl"
TSV_TRAIN_INFERENCE_FILE = "inference.train.tsv"
TEST_INFERENCE_FILE = "test.inference.pl"
TSV_TEST_INFERENCE_FILE = "inference.test.tsv"

logger = logging.getLogger(__name__)


def read_logic_file(filepath):
    """
    Reads the logic file.

    :param filepath: the filepath
    :type filepath: str
    :return: the parsed clauses
    :rtype: collections.Collection[Clause]
    """
    logger.info("Reading file:\t%s", filepath)
    start_func = performance_time()
    clauses = read_logic_program_from_file(filepath)
    end_func = performance_time()
    logger.info("\t- Reading time:\t%0.3fs", end_func - start_func)
    return clauses


def default_revision_operator():
    """
    Gets the default revision operator.

    :return: a list containing the default revision operator
    :rtype: List[RevisionOperatorEvaluator]
    """
    revision_operator = BottomClauseBoundedRule()
    revision_operator.theory_metric = RocCurveMetric()
    return [revision_operator]


def default_theory_metrics():
    """
    Gets the default theory metrics.

    :return: a list containing the default theory metrics
    :rtype: List[TheoryMetric]
    """
    return [
        RocCurveMetric(), PrecisionRecallCurveMetric(), LogLikelihoodMetric()
    ]


class StructureLearningMethod(Initializable, ABC):
    """
    Class to perform the structure learning.
    """

    OPTIONAL_FIELDS = {
        "load_pre_trained_parameter": False,
        "theory_file_paths": (),
    }

    def __init__(self, output_directory,
                 engine_system_translator,
                 load_pre_trained_parameter=None,
                 theory_file_paths=None,
                 theory_revision_manager=None,
                 theory_evaluator=None,
                 incoming_example_manager=None,
                 revision_manager=None,
                 theory_metrics=None,
                 revision_operator_selector=None,
                 revision_operator_evaluators=None,
                 ):
        """
        Creates a structure learning method

        :param output_directory: the output directory
        :type output_directory: str
        :param engine_system_translator: the engine system translator
        :type engine_system_translator: EngineSystemTranslator
        :param load_pre_trained_parameter: if it is to load the pre-trained
        parameters
        :type load_pre_trained_parameter: Optional[bool]
        :param theory_revision_manager: the theory revision manager
        :type theory_revision_manager: Optional[TheoryRevisionManager]
        :param theory_evaluator: the theory evaluator
        :type theory_evaluator: Optional[TheoryEvaluator]
        :param incoming_example_manager: the incoming example manager
        :type incoming_example_manager: IncomingExampleManager or None
        :param revision_manager: the revision manager
        :type revision_manager: Optional[RevisionManager]
        :param theory_metrics: the theory metrics
        :type theory_metrics: Optional[List[TheoryMetric]]
        :param revision_operator_selector: the revision operator selector
        :type revision_operator_selector: Optional[RevisionOperatorSelector]
        :param revision_operator_evaluators: the revision operator evaluators
        :type revision_operator_evaluators:
            Optional[List[RevisionOperatorEvaluator]]
        """
        self.time_measure = TimeMeasure()
        self.output_directory = output_directory

        self.load_pre_trained_parameter = load_pre_trained_parameter
        if load_pre_trained_parameter is None:
            self.load_pre_trained_parameter = \
                self.OPTIONAL_FIELDS["load_pre_trained_parameter"]

        self.theory_file_paths = theory_file_paths
        if theory_file_paths is None:
            self.theory_file_paths = \
                list(self.OPTIONAL_FIELDS["theory_file_paths"])

        self._knowledge_base = NeuralLogProgram()
        self._theory = NeuralLogProgram()

        self.engine_system_translator = engine_system_translator
        self.theory_revision_manager = theory_revision_manager
        self.theory_evaluator = theory_evaluator
        self.incoming_example_manager = incoming_example_manager
        self.revision_manager = revision_manager
        self.theory_metrics = theory_metrics
        self.revision_operator_selector = revision_operator_selector
        self.revision_operator_evaluators = revision_operator_evaluators

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["engine_system_translator", "output_directory"]

    # noinspection PyMissingOrEmptyDocstring
    @property
    def knowledge_base(self):
        if hasattr(self, "learning_system") and \
                self.learning_system is not None:
            return self.learning_system.knowledge_base
        else:
            return self._knowledge_base

    # noinspection PyMissingOrEmptyDocstring
    @property
    def theory(self):
        if hasattr(self, "learning_system") and \
                self.learning_system is not None:
            return self.learning_system.theory
        else:
            return self._theory

    def _build_output_directory(self):
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        if not hasattr(self, "_knowledge_base") or not self._knowledge_base:
            self._knowledge_base = NeuralLogProgram()
        if not hasattr(self, "_theory") or not self._theory:
            self._theory = NeuralLogProgram()

        self.time_measure = TimeMeasure()
        self.time_measure.add_measure(RunTimestamps.BEGIN)
        self.time_measure.add_measure(RunTimestamps.BEGIN_INITIALIZE)
        self.build()
        self.time_measure.add_measure(RunTimestamps.END_INITIALIZE)
        logger.info("%s in \t%.3fs",
                    RunTimestamps.END_INITIALIZE.value,
                    self.time_measure.time_between_timestamps(
                        RunTimestamps.BEGIN_READ_THEORY,
                        RunTimestamps.END_READ_THEORY))

    # noinspection PyMissingOrEmptyDocstring
    def run(self):
        logger.info("Running %s", self.__class__.__name__)
        self._build_output_directory()
        self.learn()
        self.evaluate_model()
        self.time_measure.add_measure(RunTimestamps.BEGIN_DISK_OUTPUT)
        self.save_model()
        self.time_measure.add_measure(RunTimestamps.END_DISK_OUTPUT)
        self.time_measure.add_measure(RunTimestamps.END)
        self.log_statistics()
        self.log_elapsed_times()
        self.save_statistics()

    @abstractmethod
    def learn(self):
        """
        Learns the model.
        """
        pass

    @abstractmethod
    def evaluate_model(self):
        """
        Evaluates the model.
        """
        pass

    def save_model(self):
        """
        Saves the model to the output directory.
        """
        self.learning_system.save_parameters(self.output_directory)

    @abstractmethod
    def log_statistics(self):
        """
        Logs the statistics of the run.
        """
        pass

    @abstractmethod
    def log_elapsed_times(self):
        """
        Logs the elapsed times of the run.
        """
        pass

    @abstractmethod
    def save_statistics(self):
        """
        Saves the statistics to the output directory.
        """
        pass

    def build(self):
        """
        Builds the learning method.
        """
        self.build_knowledge_base()
        self.build_theory()
        self.build_examples()
        self.build_statistics()
        self.build_engine_system_translator()
        self.build_learning_system()

    @abstractmethod
    def build_knowledge_base(self):
        """
        Builds the knowledge base.
        """
        pass

    # noinspection PyAttributeOutsideInit,DuplicatedCode
    def build_theory(self):
        """
        Builds the logic theory.
        """
        logger.info(RunTimestamps.BEGIN_READ_THEORY.value)
        self.time_measure.add_measure(RunTimestamps.BEGIN_READ_THEORY)
        if self.theory_file_paths:
            for filepath in self.theory_file_paths:
                clauses = read_logic_file(filepath)
                self._theory.add_clauses(clauses)
        self.time_measure.add_measure(RunTimestamps.END_READ_THEORY)
        logger.info("%s in \t%.3fs", RunTimestamps.END_READ_THEORY.value,
                    self.time_measure.time_between_timestamps(
                        RunTimestamps.BEGIN_READ_THEORY,
                        RunTimestamps.END_READ_THEORY))

    @abstractmethod
    def build_examples(self):
        """
        Builds the train examples.
        """
        pass

    @abstractmethod
    def build_statistics(self):
        """
        Builds the run statistics.
        """
        pass

    def build_engine_system_translator(self):
        """
        Builds the engine system translator.
        """
        logger.info(RunTimestamps.BEGIN_BUILD_ENGINE_TRANSLATOR.value)
        self.time_measure.add_measure(
            RunTimestamps.BEGIN_BUILD_ENGINE_TRANSLATOR)
        self.engine_system_translator.output_path = self.output_directory
        self.engine_system_translator.initialize()
        if self.load_pre_trained_parameter:
            self.engine_system_translator.load_parameters(self.output_directory)
        self.time_measure.add_measure(RunTimestamps.END_BUILD_ENGINE_TRANSLATOR)
        logger.info("%s in \t%.3fs",
                    RunTimestamps.END_BUILD_ENGINE_TRANSLATOR.value,
                    self.time_measure.time_between_timestamps(
                        RunTimestamps.BEGIN_BUILD_ENGINE_TRANSLATOR,
                        RunTimestamps.END_BUILD_ENGINE_TRANSLATOR))

    def build_learning_system(self):
        """
        Builds the learning system.
        """
        logger.info(
            "Build the learning system:\t%s", StructureLearningSystem.__name__)
        # noinspection PyAttributeOutsideInit
        self.learning_system = StructureLearningSystem(
            self._knowledge_base, self._theory, self.engine_system_translator)
        self.build_theory_metrics()
        self.build_operator_selector()

        self.build_incoming_example_manager()
        self.build_theory_evaluator()
        self.build_theory_revision_manager()

        self.learning_system.initialize()
        self.engine_system_translator.log_parameters()

    def build_theory_metrics(self):
        """
        Builds the theory metrics.
        """
        if not self.theory_metrics:
            self.theory_evaluator = default_theory_metrics()

    def build_operator_selector(self):
        """
        Builds the operator selector
        """
        if self.revision_operator_selector is None:
            self.revision_operator_selector = SelectFirstRevisionOperator()
        if not hasattr(self.revision_operator_selector, "operator_evaluators") \
                or self.revision_operator_selector.operator_evaluators is None:
            self.revision_operator_selector.operator_evaluators = \
                self.build_operators()

    def build_operators(self):
        """
        Builds the operators.

        :return: the operators
        :rtype: List[RevisionOperatorEvaluator]
        """
        if not self.revision_operator_evaluators:
            self.revision_operator_evaluators = default_revision_operator()
        else:
            self.revision_operator_evaluators = \
                list(self.revision_operator_evaluators)
        # for operator in self.revision_operator_evaluators:
        #     operator.clause_modifiers = self.clause_modifiers
        return self.revision_operator_evaluators

    def build_incoming_example_manager(self):
        """
        Builds the incoming example manager.
        """
        if self.incoming_example_manager is None:
            self.incoming_example_manager = ReviseAllIncomingExample(
                sample_selector=AllRelevantSampleSelector())
        self.learning_system.incoming_example_manager = \
            self.incoming_example_manager

    def build_theory_evaluator(self):
        """
        Builds the theory evaluator.
        """
        if self.theory_evaluator is None:
            self.theory_evaluator = TheoryEvaluator()
        self.theory_evaluator.theory_metrics = self.theory_metrics
        self.learning_system.theory_evaluator = self.theory_evaluator

    def build_theory_revision_manager(self):
        """
        Builds the revision manager.
        """
        if self.theory_revision_manager is None:
            self.theory_revision_manager = TheoryRevisionManager()
        self.build_revision_manager()

        self.theory_revision_manager.revision_manager = self.revision_manager
        self.learning_system.theory_revision_manager = \
            self.theory_revision_manager

    def build_revision_manager(self):
        """
        Builds the revision manager.
        """
        if self.revision_manager is None:
            self.revision_manager = RevisionManager()
        self.revision_manager.operator_selector = \
            self.revision_operator_selector

    def __repr__(self):
        return f"[{self.__class__.__name__}] " \
               f"output_directory: {self.output_directory}"


class BatchStructureLearning(StructureLearningMethod):
    """
    Class to learn the logic program from a batch of examples.
    """

    OPTIONAL_FIELDS = dict(StructureLearningMethod.OPTIONAL_FIELDS)
    OPTIONAL_FIELDS.update({
        "test_file_paths": (),
        "examples_batch_size": 1,
        "train_parameters_on_remaining_examples": False,
        "theory_revision_manager": None,
        "theory_evaluator": None,
        "incoming_example_manager": None,
        "revision_manager": None,
        "theory_metrics": None,
        "revision_operator_selector": None,
        "revision_operator_evaluators": None,
        "clause_modifiers": None,
    })

    def __init__(self,
                 knowledge_base_file_paths,
                 example_file_paths,
                 output_directory,
                 engine_system_translator,
                 theory_file_paths=None,
                 test_file_paths=None,
                 load_pre_trained_parameter=None,
                 examples_batch_size=None,
                 train_parameters_on_remaining_examples=None,
                 theory_revision_manager=None,
                 theory_evaluator=None,
                 incoming_example_manager=None,
                 revision_manager=None,
                 theory_metrics=None,
                 revision_operator_selector=None,
                 revision_operator_evaluators=None,
                 ):
        """
        Creates a batch structure learning method.

        :param knowledge_base_file_paths: the path of the knowledge base files
        :type knowledge_base_file_paths: list[str]
        :param example_file_paths: the path of example files
        :type example_file_paths: list[str]
        :param output_directory: the output directory
        :type output_directory: str
        :param engine_system_translator: the engine system translator
        :type engine_system_translator: EngineSystemTranslator
        :param theory_file_paths: the path of the theory files
        :type theory_file_paths: Optional[collections.Iterable[str]]
        :param test_file_paths: the path of test files
        :type test_file_paths: Optional[list[str]]
        :param load_pre_trained_parameter: if it is to load the pre-trained
        parameters
        :type load_pre_trained_parameter: Optional[bool]
        :param examples_batch_size: the example batch size
        :type examples_batch_size: Optional[int]
        :param train_parameters_on_remaining_examples: if `True`, it trains the
        parameters of the engine system translator on the remaining examples
        that were not used on the revision.
        :type train_parameters_on_remaining_examples: Optional[bool]
        """
        super(BatchStructureLearning, self).__init__(
            output_directory, engine_system_translator,
            theory_file_paths=theory_file_paths,
            load_pre_trained_parameter=load_pre_trained_parameter,
            theory_revision_manager=theory_revision_manager,
            theory_evaluator=theory_evaluator,
            incoming_example_manager=incoming_example_manager,
            revision_manager=revision_manager,
            theory_metrics=theory_metrics,
            revision_operator_selector=revision_operator_selector,
            revision_operator_evaluators=revision_operator_evaluators
        )

        self.knowledge_base_file_paths = knowledge_base_file_paths
        if theory_file_paths is None:
            self.theory_file_paths = \
                list(self.OPTIONAL_FIELDS["theory_file_paths"])
        self.example_file_paths = example_file_paths

        self.test_file_paths = test_file_paths
        if test_file_paths is None:
            self.test_file_paths = self.OPTIONAL_FIELDS["test_file_paths"]

        self.examples_batch_size = examples_batch_size
        if examples_batch_size is None:
            self.examples_batch_size = \
                self.OPTIONAL_FIELDS["examples_batch_size"]

        self.train_parameters_on_remaining_examples = \
            train_parameters_on_remaining_examples
        if train_parameters_on_remaining_examples is None:
            self.train_parameters_on_remaining_examples = \
                self.OPTIONAL_FIELDS["train_parameters_on_remaining_examples"]

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields() + ["knowledge_base_file_paths",
                                            "example_file_paths"]

    # noinspection PyAttributeOutsideInit,DuplicatedCode
    def build_knowledge_base(self):
        """
        Builds the knowledge base.
        """
        logger.info(RunTimestamps.BEGIN_READ_KNOWLEDGE_BASE.value)
        self.time_measure.add_measure(RunTimestamps.BEGIN_READ_KNOWLEDGE_BASE)
        for filepath in self.knowledge_base_file_paths:
            clauses = read_logic_file(filepath)
            self._knowledge_base.add_clauses(clauses)
        self._knowledge_base.build_program()
        self.time_measure.add_measure(RunTimestamps.END_READ_KNOWLEDGE_BASE)
        logger.info(
            "%s in \t%.3fs", RunTimestamps.END_READ_KNOWLEDGE_BASE.value,
            self.time_measure.time_between_timestamps(
                RunTimestamps.BEGIN_READ_KNOWLEDGE_BASE,
                RunTimestamps.END_READ_KNOWLEDGE_BASE))

    def build_examples(self):
        """
        Builds the train examples.
        """
        logger.info(RunTimestamps.BEGIN_READ_EXAMPLES.value)
        self.time_measure.add_measure(RunTimestamps.BEGIN_READ_EXAMPLES)
        train_examples = 0
        test_examples = 0
        for filepath in self.example_file_paths:
            clauses = read_logic_file(filepath)
            self._knowledge_base.add_clauses(clauses,
                                             example_set=TRAIN_SET_NAME)
            train_examples += len(clauses)
        for filepath in self.test_file_paths:
            clauses = read_logic_file(filepath)
            self._knowledge_base.add_clauses(
                clauses, example_set=VALIDATION_SET_NAME)
            test_examples += len(clauses)
        self.time_measure.add_measure(RunTimestamps.END_READ_EXAMPLES)
        logger.info("Total number of train examples:\t%d", train_examples)
        logger.info("Total number of test examples:\t%d", test_examples)
        logger.info("%s in \t%.3fs", RunTimestamps.END_READ_EXAMPLES.value,
                    self.time_measure.time_between_timestamps(
                        RunTimestamps.BEGIN_READ_EXAMPLES,
                        RunTimestamps.END_READ_EXAMPLES))

    # noinspection PyAttributeOutsideInit
    def build_statistics(self):
        """
        Builds the run statistics.
        """
        number_of_facts = sum(map(
            lambda x: len(x),
            self._knowledge_base.facts_by_predicate.values()))
        number_of_train = sum(map(
            lambda x: len(x),
            self._knowledge_base.examples.get(
                TRAIN_SET_NAME, {0: []}).values()))
        number_of_test = sum(map(
            lambda x: len(x),
            self._knowledge_base.examples.get(
                VALIDATION_SET_NAME, {0: []}).values()))
        self.run_statistics = \
            RunStatistics(number_of_facts, number_of_train, number_of_test)
        self.run_statistics.time_measure = self.time_measure

    # noinspection PyMissingOrEmptyDocstring
    def learn(self):
        logger.info(RunTimestamps.BEGIN_TRAIN.value)
        self.time_measure.add_measure(RunTimestamps.BEGIN_TRAIN)
        self.revise_examples()
        self.train_remaining_examples()
        self.time_measure.add_measure(RunTimestamps.END_TRAIN)
        logger.info("%s in \t%.3fs", RunTimestamps.END_TRAIN.value,
                    self.time_measure.time_between_timestamps(
                        RunTimestamps.BEGIN_TRAIN,
                        RunTimestamps.END_TRAIN))

    def revise_examples(self):
        """
        Calls the method to revise the examples
        """
        examples = self.knowledge_base.examples[TRAIN_SET_NAME]
        logger.info("Begin the revision using\t%d example(s)", examples.size())
        if self.examples_batch_size > 0:
            self.pass_batch_examples_to_revise(self.examples_batch_size)
        else:
            self.learning_system.incoming_example_manager.incoming_examples(
                ExampleIterator(examples))
        logger.info("Ended the revision")

    def train_remaining_examples(self):
        """
        Trains the parameters of the engine system translator on the
        remaining examples.
        """
        if not self.train_parameters_on_remaining_examples:
            return
        remaining_examples = \
            self.incoming_example_manager.get_remaining_examples()
        if not remaining_examples:
            return
        logger.info("Begin the training of the \t%s remaining example(s)",
                    remaining_examples.size())
        self.learning_system.train_parameters(remaining_examples)
        self.learning_system.save_trained_parameters()
        logger.info("Ended the training of the remaining example(s)")

    def pass_batch_examples_to_revise(self, number_of_examples):
        """
        Passes a `number_of_examples` to revise, at a time.
        """
        examples = self.knowledge_base.examples[TRAIN_SET_NAME]
        iterator = LimitedIterator(
            ExampleIterator(examples), number_of_examples)
        size = examples.size()
        count = 0
        while iterator.has_next:
            count = min(size, count + number_of_examples)
            iterator.reset()
            logger.debug("Passing example\t%d/%d to revise.", count, size)
            self.learning_system.incoming_example_manager.incoming_examples(
                iterator)

    # noinspection PyMissingOrEmptyDocstring
    def evaluate_model(self):
        self.time_measure.add_measure(RunTimestamps.BEGIN_EVALUATION)
        train_set = self.knowledge_base.examples[TRAIN_SET_NAME]
        inferred_examples = self.learning_system.infer_examples(train_set)
        self.run_statistics.train_evaluation = \
            self.learning_system.evaluate(train_set, inferred_examples)
        filepath = os.path.join(self.output_directory, TRAIN_INFERENCE_FILE)
        print_predictions_to_file(train_set, inferred_examples, filepath)
        test_set = self.knowledge_base.examples.get(
            VALIDATION_SET_NAME, Examples())
        if test_set:
            inferred_examples = self.learning_system.infer_examples(test_set)
            self.run_statistics.test_evaluation = \
                self.learning_system.evaluate(test_set, inferred_examples)
            filepath = os.path.join(self.output_directory, TEST_INFERENCE_FILE)
            print_predictions_to_file(test_set, inferred_examples, filepath)
        self.time_measure.add_measure(RunTimestamps.END_EVALUATION)

    # noinspection PyMissingOrEmptyDocstring
    def save_statistics(self):
        file = open(
            os.path.join(self.output_directory, STATISTICS_FILE_NAME), "w")
        yaml.dump(self.run_statistics, file)
        file.close()

    # noinspection PyMissingOrEmptyDocstring
    def log_statistics(self):
        logger.info(self.run_statistics)

    # noinspection PyMissingOrEmptyDocstring
    def log_elapsed_times(self):
        initialize_time = self.time_measure.time_between_timestamps(
            RunTimestamps.BEGIN_INITIALIZE, RunTimestamps.END_INITIALIZE)
        training_time = self.time_measure.time_between_timestamps(
            RunTimestamps.BEGIN_TRAIN, RunTimestamps.END_TRAIN)
        evaluation_time = self.time_measure.time_between_timestamps(
            RunTimestamps.BEGIN_EVALUATION, RunTimestamps.END_EVALUATION)
        output_time = self.time_measure.time_between_timestamps(
            RunTimestamps.BEGIN_DISK_OUTPUT, RunTimestamps.END_DISK_OUTPUT)
        total_time = self.time_measure.time_between_timestamps(
            RunTimestamps.BEGIN, RunTimestamps.END)

        logger.info("Total initialization time:\t%.3fs", initialize_time)
        logger.info("Total training time:\t\t%.3fs", training_time)
        logger.info("Total evaluation time:\t\t%.3fs", evaluation_time)
        logger.info("Total output time:\t\t\t%.3fs", output_time)
        logger.info("Total elapsed time:\t\t\t%.3fs", total_time)


# noinspection DuplicatedCode
def get_sorted_directories(data_directory, prefix):
    """
    Gets the iteration directories, sorted by the iteration number.

    :param data_directory: the data directory
    :type data_directory: str
    :param prefix: the iteration prefix
    :type prefix: str
    :return: the iteration directories
    :rtype: List[str]
    """
    pattern = re.compile(prefix + "([0-9]|[1-9][0-9]+)")
    directories = os.listdir(data_directory)
    directories = map(lambda x: pattern.fullmatch(x), directories)
    directories = filter(lambda x: x is not None, directories)
    directories = map(lambda x: (x.string, int(x.groups()[0])), directories)
    directories = sorted(directories, key=lambda x: x[1])
    directories = map(lambda x: x[0], directories)
    return list(directories)


def _count_logic_entries(logic_file):
    facts = 0
    rules = 0
    examples = 0
    parameters = 0

    for clause in logic_file:
        if isinstance(clause, AtomClause):
            name = clause.atom.predicate.name
            if name in NeuralLogProgram.EXAMPLES_PREDICATES:
                examples += 1
            elif name not in \
                    NeuralLogProgram.BUILTIN_PREDICATES:
                facts += 1
            else:
                parameters += 1
        else:
            rules += 1

    return facts, rules, examples, parameters


class IterativeStructureLearning(StructureLearningMethod):
    """
    Class to learn the logic program from iterations of examples.
    """

    OPTIONAL_FIELDS = dict(StructureLearningMethod.OPTIONAL_FIELDS)
    OPTIONAL_FIELDS.update({
        "iteration_prefix": "ITERATION_",
        "logic_file_extension": ".pl",
        "train_parameters_on_remaining_examples": False,
        "theory_revision_manager": None,
        "theory_evaluator": None,
        "incoming_example_manager": None,
        "revision_manager": None,
        "theory_metrics": None,
        "revision_operator_selector": None,
        "revision_operator_evaluators": None,
        "clause_modifiers": None,
    })

    def __init__(self,
                 data_directory,
                 output_directory,
                 engine_system_translator,
                 theory_file_paths=None,
                 iteration_prefix=None,
                 logic_file_extension=None,
                 load_pre_trained_parameter=None,
                 train_parameters_on_remaining_examples=None,
                 theory_revision_manager=None,
                 theory_evaluator=None,
                 incoming_example_manager=None,
                 revision_manager=None,
                 theory_metrics=None,
                 revision_operator_selector=None,
                 revision_operator_evaluators=None,
                 ):
        """
        Creates a batch structure learning method.

        :param data_directory: the path of the data directory
        :type data_directory: str
        :param output_directory: the output directory
        :type output_directory: str
        :param engine_system_translator: the engine system translator
        :type engine_system_translator: EngineSystemTranslator
        :param iteration_prefix: the iteration prefix
        :type iteration_prefix: Optional[str]
        :param logic_file_extension: the logic file extension
        :type logic_file_extension: Optional[str]
        :param load_pre_trained_parameter: if it is to load the pre-trained
        parameters
        :type load_pre_trained_parameter: Optional[bool]
        :param train_parameters_on_remaining_examples: if `True`, it trains the
        parameters of the engine system translator on the remaining examples
        that were not used on the revision.
        :type train_parameters_on_remaining_examples: Optional[bool]
        :param theory_revision_manager: the theory revision manager
        :type theory_revision_manager: Optional[TheoryRevisionManager]
        :param theory_evaluator: the theory evaluator
        :type theory_evaluator: Optional[TheoryEvaluator]
        :param incoming_example_manager: the incoming example manager
        :type incoming_example_manager: Optional[IncomingExampleManager]
        :param revision_manager: the revision manager
        :type revision_manager: Optional[RevisionManager]
        :param theory_metrics: the theory metrics
        :type theory_metrics: Optional[List[TheoryMetric]]
        :param revision_operator_selector: the revision operator selector
        :type revision_operator_selector: Optional[RevisionOperatorSelector]
        :param revision_operator_evaluators: the revision operator evaluators
        :type revision_operator_evaluators:
            Optional[List[RevisionOperatorEvaluator]]
        # :param clause_modifiers: a iterable of clause modifiers
        # :type clause_modifiers: Optional[collections.Iterable[ClauseModifier]]
        """
        super(IterativeStructureLearning, self).__init__(
            output_directory, engine_system_translator,
            theory_file_paths=theory_file_paths,
            load_pre_trained_parameter=load_pre_trained_parameter,
            theory_revision_manager=theory_revision_manager,
            theory_evaluator=theory_evaluator,
            incoming_example_manager=incoming_example_manager,
            revision_manager=revision_manager,
            theory_metrics=theory_metrics,
            revision_operator_selector=revision_operator_selector,
            revision_operator_evaluators=revision_operator_evaluators)

        self.data_directory = data_directory

        self.iteration_prefix = iteration_prefix
        if iteration_prefix is None:
            self.iteration_prefix = self.OPTIONAL_FIELDS["iteration_prefix"]

        self.logic_file_extension = logic_file_extension
        if logic_file_extension is None:
            self.logic_file_extension = \
                self.OPTIONAL_FIELDS["logic_file_extension"]

        self.train_parameters_on_remaining_examples = \
            train_parameters_on_remaining_examples
        if train_parameters_on_remaining_examples is None:
            self.train_parameters_on_remaining_examples = \
                self.OPTIONAL_FIELDS["train_parameters_on_remaining_examples"]

        self.iteration_knowledge: List[Iterable[Clause]] = []
        self.iteration_directories: List[str] = []
        self.iteration_statistics: Optional[IterationStatistics] = None
        self.time_stamp_factory: Optional[IterationTimeStampFactory] = None

        self.train_inferred_examples: Optional[ExamplesInferences] = None
        self.test_inferred_examples: Optional[ExamplesInferences] = None

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields() + ["data_directory"]

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def initialize(self):
        self.iteration_knowledge: List[Iterable[Clause]] = []
        self.iteration_directories: List[str] = []
        self.iteration_statistics: Optional[IterationStatistics] = None
        self.time_stamp_factory: Optional[IterationTimeStampFactory] = None

        self.train_inferred_examples: Optional[ExamplesInferences] = None
        self.test_inferred_examples: Optional[ExamplesInferences] = None

        super().initialize()

    # noinspection PyMissingOrEmptyDocstring
    def build_knowledge_base(self):
        logger.info("Build the knowledge from iterations")
        self.iteration_directories = get_sorted_directories(
            self.data_directory, self.iteration_prefix)
        for iteration in self.iteration_directories:
            files = os.listdir(os.path.join(self.data_directory, iteration))
            clauses = []
            facts = 0
            rules = 0
            examples = 0
            parameters = 0
            for file in files:
                filepath = os.path.join(self.data_directory, iteration, file)
                if os.path.isfile(filepath) and not file.startswith(".") \
                        and file.endswith(self.logic_file_extension):
                    logic_file = read_logic_file(filepath)
                    clauses.append(logic_file)
                    if logger.isEnabledFor(logging.DEBUG):
                        values = _count_logic_entries(logic_file)
                        facts += values[0]
                        rules += values[1]
                        examples += values[2]
                        parameters += values[3]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Knowledge read from %s: ", iteration)
                logger.debug("Number of facts:        %d", facts)
                logger.debug("Number of rules:        %d", rules)
                logger.debug("Number of examples:     %d", examples)
                logger.debug("Number of parameters:   %d", parameters)
            else:
                logger.info("Knowledge read from %s", iteration)
            self.iteration_knowledge.append(chain(*clauses))

    # noinspection PyMissingOrEmptyDocstring
    def build_examples(self):
        pass

    # noinspection PyMissingOrEmptyDocstring
    def build_statistics(self):
        self.time_stamp_factory = IterationTimeStampFactory()
        self.iteration_statistics = IterationStatistics()
        self.iteration_statistics.iteration_names = self.iteration_directories
        self.iteration_statistics.number_of_iterations = \
            len(self.iteration_knowledge)
        self.iteration_statistics.time_measure = self.time_measure

    def add_iteration_knowledge(self, index):
        """
        Adds the iteration knowledge to the learning system.

        :param index: the index of the iteration
        :type index: int
        """
        iteration_name = self.iteration_directories[index]
        iteration_knowledge = self.iteration_knowledge[index]
        total = self.knowledge_base.add_clauses(
            iteration_knowledge, example_set=iteration_name)
        self.iteration_statistics.iteration_knowledge_sizes.append(total)
        self.engine_system_translator.build_model()
        logger.debug("Added knowledge from \t%s", iteration_name)

    def remove_iteration_examples(self, index):
        """
        Removes the examples of the iteration.

        :param index: the index of the iteration
        :type index: int
        """
        iteration_name = self.iteration_directories[index]
        self.knowledge_base.examples.pop(iteration_name, None)

    # noinspection PyMissingOrEmptyDocstring
    def learn(self):
        logger.info(RunTimestamps.BEGIN_TRAIN.value)
        self.time_measure.add_measure(RunTimestamps.BEGIN_TRAIN)

        number_of_iterations = len(self.iteration_knowledge)
        logger.info(
            "Begin the revision of %d iteration(s)", number_of_iterations)
        self.add_iteration_knowledge(0)
        for i in range(number_of_iterations):
            self.revise_iteration(i)
        logger.info("Ended the revision of the iteration(s)")

        self.time_measure.add_measure(RunTimestamps.END_TRAIN)
        logger.info("%s in \t%.3fs", RunTimestamps.END_TRAIN.value,
                    self.time_measure.time_between_timestamps(
                        RunTimestamps.BEGIN_TRAIN,
                        RunTimestamps.END_TRAIN))

    # noinspection PyMissingOrEmptyDocstring
    def evaluate_model(self):
        pass

    def revise_iteration(self, index):
        """
        Revises the theory based on the iteration.

        :param index: the index of the iteration
        :type index: index
        """
        iteration_name = self.iteration_directories[index]
        begin_stamp = self.time_stamp_factory.get_time_stamp(
            IterationTimeMessage.BEGIN, iteration_name)
        examples = self.knowledge_base.examples.get(iteration_name, Examples())
        self.time_measure.add_measure(begin_stamp)
        logger.info("Begin the revision using\t%d example(s) from\t%s",
                    examples.size(), iteration_name)
        self.learning_system.incoming_example_manager.incoming_examples(
            ExampleIterator(examples))
        self.time_measure.add_measure(self.time_stamp_factory.get_time_stamp(
            IterationTimeMessage.REVISION_DONE, iteration_name))
        logger.debug(
            "Ended the revision of the example(s) from %s", iteration_name)
        self.evaluate_iteration(index)
        if index > 1:
            self.remove_iteration_examples(index - 2)
        self.save_iteration_files(index)
        end_stamp = self.time_stamp_factory.get_time_stamp(
            IterationTimeMessage.END, iteration_name)
        self.time_measure.add_measure(end_stamp)
        logger.info("Ended the revision of\t%s with time\t%0.3fs",
                    iteration_name,
                    self.time_measure.time_between_timestamps(
                        begin_stamp, end_stamp))

    def evaluate_iteration(self, index):
        """
        Evaluates the trained iteration.

        :param index: the index of the iteration
        :type index: int
        """
        iteration_name = self.iteration_directories[index]
        logger.debug("Begin the evaluation on %s", iteration_name)
        examples = self.knowledge_base.examples.get(iteration_name, Examples())
        self.train_inferred_examples = \
            self.learning_system.infer_examples(examples)
        self.iteration_statistics.add_iteration_train_evaluation(
            self.learning_system.evaluate(
                examples, self.train_inferred_examples))
        examples_size = examples.size()
        self.iteration_statistics.iteration_examples_sizes.append(examples_size)
        if len(self.iteration_statistics.iteration_knowledge_sizes) < index:
            self.iteration_statistics.iteration_knowledge_sizes -= \
                examples_size
        self.time_measure.add_measure(self.time_stamp_factory.get_time_stamp(
            IterationTimeMessage.TRAIN_EVALUATION_DONE, iteration_name))
        logger.debug("Ended the train evaluation of %s", iteration_name)
        if index + 1 < len(self.iteration_directories):
            self.add_iteration_knowledge(index + 1)
            test_iteration = self.iteration_directories[index + 1]
            examples = self.knowledge_base.examples.get(test_iteration)
            self.test_inferred_examples = \
                self.learning_system.infer_examples(examples)
            self.iteration_statistics.add_iteration_test_evaluation(
                self.learning_system.evaluate(
                    examples, self.test_inferred_examples))
            logger.debug("Ended the test evaluation of %s", iteration_name)
        else:
            self.test_inferred_examples = None
        self.time_measure.add_measure(self.time_stamp_factory.get_time_stamp(
            IterationTimeMessage.EVALUATION_DONE, iteration_name))
        logger.debug("Ended the evaluation of %s", iteration_name)

    def save_iteration_files(self, index):
        """
        Saves the iteration files, all the needed files to evaluate the
        iteration or to restore the system to the state it was at the end of
        the iteration.

        :param index: the index of the iteration
        :type index: int
        """
        iteration_name = self.iteration_directories[index]
        iteration_path = os.path.join(self.output_directory, iteration_name)
        os.makedirs(iteration_path, exist_ok=True)
        filepath = os.path.join(iteration_path, TRAIN_INFERENCE_FILE)
        tsv_filepath = os.path.join(iteration_path, TSV_TRAIN_INFERENCE_FILE)
        train_set = self.knowledge_base.examples.get(iteration_name, Examples())
        print_predictions_to_file(
            train_set, self.train_inferred_examples, filepath)
        print_tsv_file_file(
            train_set, self.train_inferred_examples, tsv_filepath)
        if self.test_inferred_examples is not None:
            test_iteration = self.iteration_directories[index + 1]
            test_set = self.knowledge_base.examples.get(
                test_iteration, Examples())
            filepath = os.path.join(iteration_path, TEST_INFERENCE_FILE)
            tsv_filepath = os.path.join(iteration_path, TSV_TEST_INFERENCE_FILE)
            print_predictions_to_file(
                test_set, self.test_inferred_examples, filepath)
            print_tsv_file_file(
                test_set, self.test_inferred_examples, tsv_filepath)
        self.learning_system.save_parameters(iteration_path)
        self.save_statistics()
        self.time_measure.add_measure(self.time_stamp_factory.get_time_stamp(
            IterationTimeMessage.SAVING_EVALUATION_DONE, iteration_name))
        logger.debug("Iteration theory saved to: %s", iteration_path)

    def save_statistics(self):
        """
        Saves the statistics to the output directory.
        """
        statistics = copy.deepcopy(self.iteration_statistics)
        statistics.time_measure = \
            self.time_measure.convert_time_measure(lambda x: x.get_message())
        file = open(
            os.path.join(self.output_directory, STATISTICS_FILE_NAME), "w")
        yaml.dump(statistics, file)
        file.close()

    # noinspection PyMissingOrEmptyDocstring
    def log_statistics(self):
        logger.info(self.iteration_statistics)

    # noinspection PyMissingOrEmptyDocstring
    def log_elapsed_times(self):
        total_init_time = self.time_measure.time_between_timestamps(
            RunTimestamps.BEGIN_INITIALIZE, RunTimestamps.END_INITIALIZE)
        total_training_time = self.time_measure.time_between_timestamps(
            RunTimestamps.BEGIN_TRAIN, RunTimestamps.END_TRAIN)
        total_output_time = self.time_measure.time_between_timestamps(
            RunTimestamps.BEGIN_DISK_OUTPUT, RunTimestamps.END_DISK_OUTPUT)
        total_running_time = self.time_measure.time_between_timestamps(
            RunTimestamps.BEGIN, RunTimestamps.END)

        logger.info("Total initialization time:\t%0.3fs", total_init_time)
        logger.info("Total training time:\t\t%0.3fs", total_training_time)

        self.log_detailed_time_by_steps()

        logger.info("Total output time:\t\t\t%0.3fs", total_output_time)
        logger.info("Total elapsed time:\t\t\t%0.3fs", total_running_time)

    def log_detailed_time_by_steps(self):
        """
        A detailed containing the total time of the iterations by each step.
        The steps are: revision, evaluation and save outputs.
        """
        revision_time = 0
        inference_time = 0
        output_time = 0

        for i in range(len(self.iteration_knowledge)):
            iteration_name = self.iteration_directories[i]
            revision_time += self.time_measure.time_between_timestamps(
                self.time_stamp_factory.get_time_stamp(
                    IterationTimeMessage.BEGIN, iteration_name),
                self.time_stamp_factory.get_time_stamp(
                    IterationTimeMessage.REVISION_DONE, iteration_name),
            )
            inference_time += self.time_measure.time_between_timestamps(
                self.time_stamp_factory.get_time_stamp(
                    IterationTimeMessage.REVISION_DONE, iteration_name),
                self.time_stamp_factory.get_time_stamp(
                    IterationTimeMessage.EVALUATION_DONE, iteration_name),
            )
            output_time += self.time_measure.time_between_timestamps(
                self.time_stamp_factory.get_time_stamp(
                    IterationTimeMessage.EVALUATION_DONE, iteration_name),
                self.time_stamp_factory.get_time_stamp(
                    IterationTimeMessage.END, iteration_name),
            )

        logger.info("\t- Iterations revision time:\t\t\t%0.3fs", revision_time)
        logger.info("\t- Iterations evaluation time:\t\t%0.3fs", inference_time)
        logger.info("\t- Iteration saving files time:\t\t%0.3fs", output_time)
