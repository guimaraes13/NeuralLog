"""
The methods to call the structure learning.
"""
import logging
import os
from abc import abstractmethod
from enum import Enum
from typing import List

import yaml

from src.knowledge.examples import Examples, ExampleIterator, LimitedIterator
from src.knowledge.manager.example_manager import ReviseAllIncomingExample
from src.knowledge.program import NeuralLogProgram
from src.knowledge.theory.evaluation.metric.theory_metric import TheoryMetric, \
    RocCurveMetric, PrecisionRecallCurveMetric, LogLikelihoodMetric
from src.knowledge.theory.evaluation.theory_evaluator import TheoryEvaluator
from src.knowledge.theory.manager.revision.operator.revision_operator import \
    BottomClauseBoundedRule
from src.knowledge.theory.manager.revision.revision_manager import \
    RevisionManager
from src.knowledge.theory.manager.revision.revision_operator_evaluator import \
    RevisionOperatorEvaluator
from src.knowledge.theory.manager.revision.revision_operator_selector import \
    SelectFirstRevisionOperator
from src.knowledge.theory.manager.revision.sample_selector import \
    AllRelevantSampleSelector
from src.knowledge.theory.manager.theory_revision_manager import \
    TheoryRevisionManager
from src.run.command import TRAIN_SET_NAME, VALIDATION_SET_NAME
from src.structure_learning.engine_system_translator import \
    EngineSystemTranslator
from src.structure_learning.structure_learning_system import \
    StructureLearningSystem
from src.util import Initializable
from src.util.file import read_logic_program_from_file, \
    print_predictions_to_file
from src.util.statistics import RunStatistics
from src.util.time_measure import TimeMeasure, performance_time

STATISTICS_FILE_NAME = "statistics.yaml"

TRAIN_INFERENCE_FILE = "train.inference.pl"
TEST_INFERENCE_FILE = "test.inference.pl"

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


class StructureLearningMethod(Initializable):
    """
    Class to perform the structure learning.
    """

    OPTIONAL_FIELDS = {"theory_file_paths": ()}

    def __init__(self,
                 knowledge_base_file_paths,
                 example_file_paths,
                 output_directory,
                 theory_file_paths=None):
        """
        Creates an structure learning method.

        :param knowledge_base_file_paths: the path of the knowledge base files
        :type knowledge_base_file_paths: list[str]
        :param theory_file_paths: the path of the theory files
        :type theory_file_paths: Optional[collections.Iterable[str]]
        :param example_file_paths: the path of example files
        :type example_file_paths: list[str]
        :param output_directory: the output directory
        :type output_directory: str
        """
        self.time_measure = TimeMeasure()
        self.time_measure.add_measure(RunTimestamps.BEGIN)
        self.knowledge_base_file_paths = knowledge_base_file_paths
        if not theory_file_paths:
            self.theory_file_paths = \
                list(self.OPTIONAL_FIELDS["theory_file_paths"])
        self.example_file_paths = example_file_paths
        self.output_directory = output_directory

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return [
            "knowledge_base_file_paths", "example_file_paths",
            "output_directory"
        ]

    @abstractmethod
    def run(self):
        """
        Runs the structure learning method.
        """
        pass


class BatchStructureLearning(StructureLearningMethod):
    """
    Class to learn the logic program from a batch of examples.
    """

    OPTIONAL_FIELDS = {
        "theory_file_paths": (),
        "test_file_paths": (),
        "load_pre_trained_parameter": False,
        "examples_batch_size": 1,
        "pass_all_examples_at_once": False,
        "train_parameters_on_remaining_examples": False,
        "theory_revision_manager": None,
        "theory_evaluator": None,
        "incoming_example_manager": None,
        "revision_manager": None,
        "theory_metrics": None,
        "revision_operator_selector": None,
        "revision_operator_evaluators": None,
        "clause_modifiers": None,
    }

    def __init__(self,
                 knowledge_base_file_paths,
                 example_file_paths,
                 output_directory,
                 engine_system_translator,
                 theory_file_paths=None,
                 test_file_paths=None,
                 load_pre_trained_parameter=None,
                 examples_batch_size=None,
                 pass_all_examples_at_once=None,
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
        :param pass_all_examples_at_once: if `True`, all the examples will be
        passed to be revised at once; otherwise, will be passed a batch at a
        time
        :type pass_all_examples_at_once: Optional[bool]
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
        super(BatchStructureLearning, self).__init__(
            knowledge_base_file_paths,
            example_file_paths,
            output_directory,
            theory_file_paths
        )

        self.engine_system_translator = engine_system_translator

        self.theory_file_paths = theory_file_paths
        if not theory_file_paths:
            self.theory_file_paths = \
                list(self.OPTIONAL_FIELDS["theory_file_paths"])

        self.test_file_paths = test_file_paths
        if not test_file_paths:
            self.test_file_paths = self.OPTIONAL_FIELDS["test_file_paths"]

        self.load_pre_trained_parameter = load_pre_trained_parameter
        if not load_pre_trained_parameter:
            self.load_pre_trained_parameter = \
                self.OPTIONAL_FIELDS["load_pre_trained_parameter"]

        self.examples_batch_size = examples_batch_size
        if not examples_batch_size:
            self.examples_batch_size = \
                self.OPTIONAL_FIELDS["examples_batch_size"]

        self.pass_all_examples_at_once = pass_all_examples_at_once
        if not pass_all_examples_at_once:
            self.pass_all_examples_at_once = \
                self.OPTIONAL_FIELDS["pass_all_examples_at_once"]

        self.train_parameters_on_remaining_examples = \
            train_parameters_on_remaining_examples
        if not train_parameters_on_remaining_examples:
            self.train_parameters_on_remaining_examples = \
                self.OPTIONAL_FIELDS["train_parameters_on_remaining_examples"]

        self.theory_revision_manager = theory_revision_manager
        self.theory_evaluator = theory_evaluator
        self.incoming_example_manager = incoming_example_manager
        self.revision_manager = revision_manager
        self.theory_metrics = theory_metrics
        self.revision_operator_selector = revision_operator_selector
        self.revision_operator_evaluators = revision_operator_evaluators

    # noinspection PyMissingOrEmptyDocstring
    @property
    def knowledge_base(self):
        if self.learning_system is not None:
            return self.learning_system.knowledge_base
        else:
            return self._knowledge_base

    # noinspection PyMissingOrEmptyDocstring
    @property
    def theory(self):
        if self.learning_system is not None:
            return self.learning_system.theory
        else:
            return self._theory

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields() + ["engine_system_translator"]

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def initialize(self):
        super().initialize()
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

    def build(self):
        """
        Builds the learning method.
        """
        self.build_knowledge_base()
        self.build_theory()
        self.build_examples()
        self.build_run_statistics()
        self.build_engine_system_translator()
        self.build_learning_system()

    # noinspection PyAttributeOutsideInit,DuplicatedCode
    def build_knowledge_base(self):
        """
        Builds the knowledge base.
        """
        logger.info(RunTimestamps.BEGIN_READ_KNOWLEDGE_BASE.value)
        self.time_measure.add_measure(RunTimestamps.BEGIN_READ_KNOWLEDGE_BASE)
        self._knowledge_base = NeuralLogProgram()
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

    # noinspection PyAttributeOutsideInit,DuplicatedCode
    def build_theory(self):
        """
        Builds the logic theory.
        """
        logger.info(RunTimestamps.BEGIN_READ_THEORY.value)
        self.time_measure.add_measure(RunTimestamps.BEGIN_READ_THEORY)
        self._theory = NeuralLogProgram()
        if self.theory_file_paths:
            for filepath in self.theory_file_paths:
                clauses = read_logic_file(filepath)
                self._theory.add_clauses(clauses)
        self.time_measure.add_measure(RunTimestamps.END_READ_THEORY)
        logger.info("%s in \t%.3fs", RunTimestamps.END_READ_THEORY.value,
                    self.time_measure.time_between_timestamps(
                        RunTimestamps.BEGIN_READ_THEORY,
                        RunTimestamps.END_READ_THEORY))

    def build_examples(self):
        """
        Builds the train examples.
        """
        logger.info(RunTimestamps.BEGIN_READ_EXAMPLES.name)
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
    def build_run_statistics(self):
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

    def build_engine_system_translator(self):
        """
        Builds the engine system translator.
        """
        logger.info(RunTimestamps.BEGIN_BUILD_ENGINE_TRANSLATOR.name)
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

    # noinspection PyMissingOrEmptyDocstring
    def run(self):
        logger.info("Running %s", self.__class__.__name__)
        self.learn()
        self._build_output_directory()
        self.evaluate_model()
        self.time_measure.add_measure(RunTimestamps.BEGIN_DISK_OUTPUT)
        self.save_model()
        self.time_measure.add_measure(RunTimestamps.END_DISK_OUTPUT)
        self.time_measure.add_measure(RunTimestamps.END)
        logger.info(self.run_statistics)
        self.log_elapsed_times()
        self.save_statistics()

    def _build_output_directory(self):
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

    def learn(self):
        """
        Learns the model.
        """
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
        examples = self._knowledge_base.examples[TRAIN_SET_NAME]
        logger.info("Begin the revision using\t%s example(s)", examples.size())
        if self.pass_all_examples_at_once:
            self.learning_system.incoming_example_manager.incoming_examples(
                ExampleIterator(examples))
        elif self.examples_batch_size > 1:
            self.pass_batch_examples_to_revise(self.examples_batch_size)
        else:
            self.pass_batch_examples_to_revise(1)
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
        examples = self._knowledge_base.examples[TRAIN_SET_NAME]
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

    def evaluate_model(self):
        """
        Evaluates the model.
        """
        self.time_measure.add_measure(RunTimestamps.BEGIN_EVALUATION)
        train_set = self._knowledge_base.examples[TRAIN_SET_NAME]
        inferred_examples = self.learning_system.infer_examples(train_set)
        self.run_statistics.train_evaluation = \
            self.learning_system.evaluate(train_set, inferred_examples)
        filepath = os.path.join(self.output_directory, TRAIN_INFERENCE_FILE)
        print_predictions_to_file(train_set, inferred_examples, filepath)
        test_set = self._knowledge_base.examples.get(
            VALIDATION_SET_NAME, Examples())
        if test_set:
            inferred_examples = self.learning_system.infer_examples(test_set)
            self.run_statistics.test_evaluation = \
                self.learning_system.evaluate(test_set, inferred_examples)
            filepath = os.path.join(self.output_directory, TEST_INFERENCE_FILE)
            print_predictions_to_file(test_set, inferred_examples, filepath)
        self.time_measure.add_measure(RunTimestamps.END_EVALUATION)

    def save_model(self):
        """
        Saves the model to the output directory.
        """
        self.learning_system.save_parameters(self.output_directory)

    def save_statistics(self):
        """
        Saves the statistics to the output directory.
        """
        file = open(
            os.path.join(self.output_directory, STATISTICS_FILE_NAME), "w")
        yaml.dump(self.run_statistics, file)
        file.close()

    def log_elapsed_times(self):
        """
        Logs the elapsed times of the run.
        """
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

    def __repr__(self):
        description = \
            "\tSettings:\n" + \
            "\tKnowledge base files:\t" + \
            ", ".join(self.knowledge_base_file_paths or []) + "\n" + \
            "\t Theory files:\t\t\t" + \
            ", ".join(self.theory_file_paths or []) + "\n" + \
            "\tExample files:\t\t\t" + \
            ", ".join(self.example_file_paths or []) + "\n" + \
            "\tTest files:\t\t\t\t" + \
            ", ".join(self.test_file_paths or [])

        return description.strip()


class RunTimestamps(Enum):
    """
    A enumerator of the names for run timestamps.
    """

    BEGIN = "Begin"
    BEGIN_INITIALIZE = "Begin initializing"
    END_INITIALIZE = "End initializing"
    BEGIN_READ_KNOWLEDGE_BASE = "Begin reading the knowledge base"
    END_READ_KNOWLEDGE_BASE = "End reading the knowledge base"
    BEGIN_READ_THEORY = "Begin reading the theory"
    END_READ_THEORY = "End reading the theory"
    BEGIN_READ_EXAMPLES = "Begin reading the examples"
    END_READ_EXAMPLES = "End reading the examples"
    BEGIN_BUILD_ENGINE_TRANSLATOR = "Begin building the engine translator"
    END_BUILD_ENGINE_TRANSLATOR = "End building the engine translator"
    BEGIN_TRAIN = "Begin training"
    END_TRAIN = "End training"
    BEGIN_EVALUATION = "Begin evaluating"
    END_EVALUATION = "End evaluating"
    BEGIN_DISK_OUTPUT = "Begin disk output"
    END_DISK_OUTPUT = "End disk output"
    END = "Finished"
