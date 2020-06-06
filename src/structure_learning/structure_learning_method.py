"""
The methods to call the structure learning.
"""
import logging
from abc import abstractmethod

from src.knowledge.manager.example_manager import IncomingExampleManager
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
from src.structure_learning.engine_system_translator import \
    EngineSystemTranslator
from src.util import Initializable
from src.util.statistics import RunStatistics
from src.util.time.time_measure import TimeMeasure, RunTimestamps

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
    def run(self) -> None:
        """
        Runs the structure learning method.
        """
        pass


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
    def initialize(self) -> None:
        self.time_measure.add_measure(RunTimestamps.BEGIN_INITIALIZE)
        logger.info("Initializing %s", self.__class__.__name__)
        super().initialize()
        self.build()
        self.run_statistics = RunStatistics(0, 0, 0)
        self.time_measure.add_measure(RunTimestamps.END_INITIALIZE)

    def build(self):
        """
        Builds the learning method.
        """
        self.build_knowledge_base()
        self.build_theory()
        self.build_examples()
        self.build_engine_system_translator()
        self.build_learning_system()

    def build_knowledge_base(self):
        """
        Builds the knowledge base.
        """
        pass

    def build_theory(self):
        """
        Builds the logic theory.
        """
        pass

    def build_examples(self):
        """
        Builds the train examples.
        """
        pass

    def build_engine_system_translator(self):
        """
        Builds the engine system translator.
        """
        pass

    def build_learning_system(self):
        """
        Builds the learning system.
        """
        pass

    # noinspection PyMissingOrEmptyDocstring
    def run(self) -> None:
        logger.info("Running %s", self.__class__.__name__)
        self.train()
        self.time_measure.add_measure(RunTimestamps.BEGIN_DISK_OUTPUT)
        self.save_model()
        self.save_statistics()
        self.time_measure.add_measure(RunTimestamps.END_DISK_OUTPUT)
        self.time_measure.add_measure(RunTimestamps.END)

    def train(self):
        """
        Trains the model.
        """
        logger.info("Begin training")
        self.time_measure.add_measure(RunTimestamps.BEGIN_TRAIN)

        self.time_measure.add_measure(RunTimestamps.END_TRAIN)

    def save_model(self):
        """
        Saves the model to the output directory.
        """
        pass

    def save_statistics(self):
        """
        Saves the statistics to the output directory.
        """
        pass
