"""
The methods to call the structure learning.
"""
import logging
from abc import abstractmethod
from enum import Enum

from src.knowledge.manager.example_manager import IncomingExampleManager
from src.knowledge.program import NeuralLogProgram
from src.knowledge.theory.evaluation.metric.theory_metric import TheoryMetric
from src.knowledge.theory.evaluation.theory_evaluator import TheoryEvaluator
from src.knowledge.theory.manager.revision.revision_manager import \
    RevisionManager
from src.knowledge.theory.manager.revision.revision_operator_evaluator import \
    RevisionOperatorEvaluator
from src.knowledge.theory.manager.revision.revision_operator_selector import \
    RevisionOperatorSelector
from src.knowledge.theory.manager.theory_revision_manager import \
    TheoryRevisionManager
from src.run.command import TRAIN_SET_NAME, VALIDATION_SET_NAME
from src.structure_learning.engine_system_translator import \
    EngineSystemTranslator
from src.util import Initializable
from src.util.file import read_logic_program_from_file
from src.util.statistics import RunStatistics
from src.util.time_measure import TimeMeasure, performance_time

logger = logging.getLogger(__name__)


class StructureLearningMethod(Initializable):
    """
    Class to perform the structure learning.
    """

    def __init__(self,
                 knowledge_base_file_paths,
                 theory_file_paths,
                 example_file_paths,
                 output_directory):
        """
        Creates an structure learning method.

        :param knowledge_base_file_paths: the path of the knowledge base files
        :type knowledge_base_file_paths: list[str]
        :param theory_file_paths: the path of the theory files
        :type theory_file_paths: list[str]
        :param example_file_paths: the path of example files
        :type example_file_paths: list[str]
        :param output_directory: the output directory
        :type output_directory: str
        """
        self.time_measure = TimeMeasure()
        self.time_measure.add_measure(RunTimestamps.BEGIN)
        self.knowledge_base_file_paths = knowledge_base_file_paths
        self.theory_file_paths = theory_file_paths
        self.example_file_paths = example_file_paths
        self.output_directory = output_directory

    @abstractmethod
    def run(self):
        """
        Runs the structure learning method.
        """
        pass


def read_logic_file(filepath):
    """
    Reads the logic file.

    :param filepath: the filepath
    :type filepath: str
    :return: the parsed clauses
    :rtype: collections.Iterable[Clause]
    """
    logger.info("Reading file:\t%s", filepath)
    start_func = performance_time()
    clauses = read_logic_program_from_file(filepath)
    end_func = performance_time()
    logger.info("\t- Reading time:\t%0.3fs", end_func - start_func)
    return clauses


class BatchStructureLearning(StructureLearningMethod):
    """
    Class to learn the logic program from a batch of examples.
    """

    def __init__(self,
                 knowledge_base_file_paths,
                 theory_file_paths,
                 example_file_paths,
                 test_file_paths,
                 output_directory,
                 load_pre_trained_parameter,
                 examples_batch_size,
                 engine_system_translator,
                 theory_revision_manager,
                 theory_evaluator,
                 incoming_example_manager,
                 revision_manager,
                 theory_metrics,
                 revision_operator_selector,
                 revision_operator_evaluators,
                 ):
        """
        Creates a batch structure learning method.

        :param knowledge_base_file_paths: the path of the knowledge base files
        :type knowledge_base_file_paths: list[str]
        :param theory_file_paths: the path of the theory files
        :type theory_file_paths: list[str]
        :param example_file_paths: the path of example files
        :type example_file_paths: list[str]
        :param test_file_paths: the path of test files
        :type test_file_paths: list[str]
        :param output_directory: the output directory
        :type output_directory: str
        :param load_pre_trained_parameter: if it is to load the pre-trained
        parameters
        :type load_pre_trained_parameter: bool
        :param examples_batch_size: the example batch size
        :type examples_batch_size: int
        :param engine_system_translator: the engine system translator
        :type engine_system_translator: EngineSystemTranslator
        :param theory_revision_manager: the theory revision manager
        :type theory_revision_manager: TheoryRevisionManager
        :param theory_evaluator: the theory evaluator
        :type theory_evaluator: TheoryEvaluator
        :param incoming_example_manager: the incoming example manager
        :type incoming_example_manager: IncomingExampleManager
        :param revision_manager: the revision manager
        :type revision_manager: RevisionManager
        :param theory_metrics: the theory metrics
        :type theory_metrics: list[TheoryMetric]
        :param revision_operator_selector: the revision operator selector
        :type revision_operator_selector: RevisionOperatorSelector
        :param revision_operator_evaluators: the revision operator evaluators
        :type revision_operator_evaluators: list[RevisionOperatorEvaluator]
        """
        super(BatchStructureLearning, self).__init__(
            knowledge_base_file_paths,
            theory_file_paths,
            example_file_paths,
            output_directory
        )
        self.test_file_paths = test_file_paths
        self.load_pre_trained_parameter = load_pre_trained_parameter
        self.examples_batch_size = examples_batch_size
        self.engine_system_translator = engine_system_translator
        self.theory_revision_manager = theory_revision_manager
        self.theory_evaluator = theory_evaluator
        self.incoming_example_manager = incoming_example_manager
        self.revision_manager = revision_manager
        self.theory_metrics = theory_metrics
        self.revision_operator_selector = revision_operator_selector
        self.revision_operator_evaluators = revision_operator_evaluators

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def initialize(self):
        super().initialize()
        self.time_measure.add_measure(RunTimestamps.BEGIN_INITIALIZE)
        super().initialize()
        self.build()
        self.time_measure.add_measure(RunTimestamps.END_INITIALIZE)
        logger.info("%s in \t%s",
                    RunTimestamps.END_INITIALIZE.value,
                    self.time_measure.timestamps(
                        RunTimestamps.BEGIN_READ_THEORY,
                        RunTimestamps.END_READ_THEORY))

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        fields = []
        for field in self.__dict__.keys():
            if field.endswith("paths"):
                continue
            fields.append(field)
        return fields

    def build(self):
        """
        Builds the learning method.
        """
        self.build_knowledge_base()
        self.build_theory()
        self.build_examples()
        self.knowledge_base.build_program()
        self.theory.build_program()
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
        self.knowledge_base = NeuralLogProgram()
        for filepath in self.knowledge_base_file_paths:
            clauses = read_logic_file(filepath)
            self.knowledge_base.add_clauses(clauses)
        self.time_measure.add_measure(RunTimestamps.END_READ_KNOWLEDGE_BASE)
        logger.info("%s in \t%s", RunTimestamps.END_READ_KNOWLEDGE_BASE.value,
                    self.time_measure.timestamps(
                        RunTimestamps.BEGIN_READ_KNOWLEDGE_BASE,
                        RunTimestamps.END_READ_KNOWLEDGE_BASE))

    # noinspection PyAttributeOutsideInit,DuplicatedCode
    def build_theory(self):
        """
        Builds the logic theory.
        """
        logger.info(RunTimestamps.BEGIN_READ_THEORY.value)
        self.time_measure.add_measure(RunTimestamps.BEGIN_READ_THEORY)
        self.theory = NeuralLogProgram()
        for filepath in self.theory_file_paths:
            clauses = read_logic_file(filepath)
            self.theory.add_clauses(clauses)
        self.time_measure.add_measure(RunTimestamps.END_READ_THEORY)
        logger.info("%s in \t%s", RunTimestamps.END_READ_THEORY.value,
                    self.time_measure.timestamps(
                        RunTimestamps.BEGIN_READ_THEORY,
                        RunTimestamps.END_READ_THEORY))

    def build_examples(self):
        """
        Builds the train examples.
        """
        logger.info(RunTimestamps.BEGIN_READ_EXAMPLES.name)
        self.time_measure.add_measure(RunTimestamps.BEGIN_READ_EXAMPLES)
        for filepath in self.example_file_paths:
            clauses = read_logic_file(filepath)
            self.knowledge_base.add_clauses(clauses, example_set=TRAIN_SET_NAME)
        for filepath in self.test_file_paths:
            clauses = read_logic_file(filepath)
            self.knowledge_base.add_clauses(
                clauses, example_set=VALIDATION_SET_NAME)
        self.time_measure.add_measure(RunTimestamps.END_READ_EXAMPLES)
        logger.info("%s in \t%s", RunTimestamps.END_READ_EXAMPLES.value,
                    self.time_measure.timestamps(
                        RunTimestamps.BEGIN_READ_EXAMPLES,
                        RunTimestamps.END_READ_EXAMPLES))

    # noinspection PyAttributeOutsideInit
    def build_run_statistics(self):
        """
        Builds the run statistics.
        """
        number_of_facts = sum(map(
            lambda x: len(x),
            self.knowledge_base.facts_by_predicate.values()))
        number_of_train = sum(map(
            lambda x: len(x),
            self.knowledge_base.examples.get(TRAIN_SET_NAME, {0: []}).values()))
        number_of_test = sum(map(
            lambda x: len(x),
            self.knowledge_base.examples.get(TRAIN_SET_NAME, {0: []}).values()))
        self.run_statistics = \
            RunStatistics(number_of_facts, number_of_train, number_of_test)

    def build_engine_system_translator(self):
        """
        Builds the engine system translator.
        """
        logger.info(RunTimestamps.BEGIN_BUILD_ENGINE_TRANSLATOR.name)
        self.time_measure.add_measure(
            RunTimestamps.BEGIN_BUILD_ENGINE_TRANSLATOR)
        program = self.knowledge_base.copy()
        for clauses in program.clauses_by_predicate.values():
            program.add_clauses(clauses)
        program.build_program()
        self.engine_system_translator.program = program
        self.engine_system_translator.initialize()
        self.time_measure.add_measure(RunTimestamps.END_BUILD_ENGINE_TRANSLATOR)
        logger.info("%s in \t%s",
                    RunTimestamps.END_BUILD_ENGINE_TRANSLATOR.value,
                    self.time_measure.timestamps(
                        RunTimestamps.BEGIN_BUILD_ENGINE_TRANSLATOR,
                        RunTimestamps.END_BUILD_ENGINE_TRANSLATOR))

    def build_learning_system(self):
        """
        Builds the learning system.
        """
        # TODO: to build the learning system
        pass

    # noinspection PyMissingOrEmptyDocstring
    def run(self):
        logger.info("Running %s", self.__class__.__name__)
        self.learn()
        self.time_measure.add_measure(RunTimestamps.BEGIN_DISK_OUTPUT)
        self.save_model()
        self.save_statistics()
        self.time_measure.add_measure(RunTimestamps.END_DISK_OUTPUT)
        self.time_measure.add_measure(RunTimestamps.END)

    def learn(self):
        """
        Learns the model.
        """
        logger.info(RunTimestamps.BEGIN_TRAIN.value)
        self.time_measure.add_measure(RunTimestamps.BEGIN_TRAIN)
        # TODO: to learn the structure
        logger.info("%s in \t%s", RunTimestamps.END_TRAIN,
                    self.time_measure.timestamps(
                        RunTimestamps.BEGIN_TRAIN,
                        RunTimestamps.END_TRAIN))

    def save_model(self):
        """
        Saves the model to the output directory.
        """
        # TODO: to save the model
        pass

    def save_statistics(self):
        """
        Saves the statistics to the output directory.
        """
        # TODO: to save the statistics and running time
        pass


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
