"""
Learns the logic program structure for a given task.
"""
import argparse
import logging
import os
from datetime import datetime
from shutil import copyfile

import yaml

import resources
import src.knowledge.manager.tree_manager
import src.knowledge.theory.manager.revision.operator.tree_revision_operator
import src.knowledge.theory.manager.revision.operator.meta_revision_operator
from src.run import configure_log
from src.run.command import Command, command, create_log_file
from src.structure_learning.structure_learning_method import \
    BatchStructureLearning
from src.util import print_args

YAML_SHORT_OPTION = "-y"
YAML_LONG_OPTION = "--yaml"

# noinspection SpellCheckingInspection
RUN_FOLDER_TIMESTAMP_FORMAT = "RUN_%Y_%m_%d_%Hh%Mmin%Ss%fms"

OUTPUT_RUN_FILE_NAME = "run.sh"
OUTPUT_CONFIGURATION_YAML = "configuration.yaml"
OUTPUT_LOG_FILE = "log.txt"

STRUCTURE_LEARNING_CLASS = BatchStructureLearning

DEFAULT_YAML_CONFIGURATION_FILE = \
    os.path.join(resources.RESOURCE_PATH, "configuration.yaml")
COMMAND_NAME = "learn_structure"

LOG_FORMAT = "[ %(asctime)s ] [ %(levelname)8s ] [ %(name)s ] - \t%(message)s"

logger = logging.getLogger(__name__)


def load_yaml_configuration(yaml_path):
    """
    Loads the structure learning method from the yaml file configuration.

    :param yaml_path: the path of the yaml file
    :type yaml_path: str
    :return: the structure learning method
    :rtype: BatchStructureLearning
    """
    stream = open(yaml_path, 'r')
    learning_method = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    return learning_method


@command(COMMAND_NAME)
class LearnStructure(Command):
    """
    Learns the logic program structure.
    """

    def __init__(self, program, args, direct=False):
        # configure_log(LOG_FORMAT, level=logging.INFO)
        super().__init__(program, args, direct)

    # noinspection PyMissingOrEmptyDocstring
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
        parser.add_argument("--knowledgeBase", "-k", metavar="kb.pl",
                            type=str, required=False, default=None, nargs="*",
                            help="The knowledge base file(s).")
        parser.add_argument("--theory", "-t", metavar="theory.pl",
                            type=str, required=False, default=None, nargs="*",
                            help="The theory file(s).")
        parser.add_argument("--examples", "-e", metavar="examples.pl",
                            type=str, required=False, default=None, nargs="*",
                            help="The examples file(s).")
        parser.add_argument("--test", "-test", metavar="test.pl",
                            type=str, required=False, default=None, nargs="*",
                            help="The test file(s).")

        parser.add_argument(YAML_LONG_OPTION, YAML_SHORT_OPTION,
                            metavar="configuration.yaml",
                            type=str, default=None, required=False,
                            help="The yaml configuration file. The yaml "
                                 "configuration file is used to set all the "
                                 "options of the system. If it is not "
                                 "provided, a default configuration file will "
                                 "be used. The command line options override "
                                 "the ones in the yaml file.")

        # Output
        parser.add_argument("--outputDirectory", "-o", metavar="output",
                            type=str, default=None, required=False,
                            help="The directory in which to save the files. "
                                 "If not specified, a new directory, in the "
                                 "current directory, will be created. "
                                 "This option creates a folder inside the "
                                 "output directory, with the name of the "
                                 "target relation and a timestamp.")
        parser.add_argument("--strictOutputDirectory", "-strict",
                            dest="strictOutput", action="store_true",
                            help="If set, the output will be saved strict to "
                                 "the output directory, without creating any "
                                 "subdirectory. This option might override "
                                 "previous files.")
        parser.set_defaults(strictOutput=False)

        # Log
        parser.add_argument("--verbose", "-v",
                            dest="verbose", action="store_true",
                            help="If set, increases the logging verbosity.")
        parser.set_defaults(verbose=False)
        # Log file is always on
        # parser.add_argument("--logFile", "-log", metavar='file',
        #                     type=str, default=None,
        #                     help="The file path to save the log into")

        return parser

    # noinspection PyAttributeOutsideInit,PyMissingOrEmptyDocstring
    def parse_args(self):
        args = self.parser.parse_args(self.args)
        level = logging.DEBUG if args.verbose else logging.INFO
        configure_log(LOG_FORMAT, level=level)
        super().parse_args()

        self.yaml_path = args.yaml
        self.yaml_option_index = None
        if self.yaml_path is None:
            self.yaml_path = DEFAULT_YAML_CONFIGURATION_FILE
        else:
            if YAML_LONG_OPTION in self.args:
                self.yaml_option_index = self.args.index(YAML_LONG_OPTION)
            elif YAML_SHORT_OPTION in self.args:
                self.yaml_option_index = self.args.index(YAML_SHORT_OPTION)

        self.learning_method = load_yaml_configuration(self.yaml_path)
        if not isinstance(self.learning_method, STRUCTURE_LEARNING_CLASS):
            logger.error(
                "An YAML configuration for a %s class was excepted, "
                "but a %s class was found.",
                STRUCTURE_LEARNING_CLASS.__name__,
                self.learning_method.__class__.__name__
            )
            return

        if args.knowledgeBase is not None:
            self.learning_method.knowledge_base_file_paths = args.knowledgeBase

        if args.theory is not None:
            self.learning_method.theory_file_paths = args.theory

        if args.examples is not None:
            self.learning_method.example_file_paths = args.examples

        if args.test is not None:
            self.learning_method.test_file_paths = args.test

        strict_output = args.strictOutput
        self.output_directory = args.outputDirectory
        if self.output_directory is None:
            if self.learning_method.output_directory is None:
                self.output_directory = os.getcwd()
            else:
                self.output_directory = self.learning_method.output_directory
                strict_output = True

        if not strict_output:
            folder = datetime.now().strftime(RUN_FOLDER_TIMESTAMP_FORMAT)
            self.output_directory = os.path.join(self.output_directory, folder)

        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

        if self.learning_method.output_directory is None:
            self.learning_method.output_directory = self.output_directory
        self.save_configuration()
        src.util.print_args(args, logger)

    def save_configuration(self):
        """
        Saves the configuration files to rerun the experiment.
        """
        create_log_file(os.path.join(self.output_directory, OUTPUT_LOG_FILE))
        run_file = open(
            os.path.join(self.output_directory, OUTPUT_RUN_FILE_NAME), "w")
        run_file.write(self.program)
        run_file.write(" ")
        run_file.write(COMMAND_NAME)

        yaml_config = os.path.join(self.output_directory,
                                   OUTPUT_CONFIGURATION_YAML)
        args = self.args
        if self.yaml_option_index is not None:
            args[self.yaml_option_index + 1] = yaml_config
        else:
            args += [YAML_LONG_OPTION, yaml_config]
        if len(args) > 0:
            run_file.write(" ")
            run_file.write(" ".join(args))
        run_file.write("\n")
        run_file.close()

        copyfile(self.yaml_path, yaml_config)

    # noinspection PyMissingOrEmptyDocstring
    def run(self):
        logger.info("Learning method:\n%s", yaml.dump(self.learning_method))
        self.learning_method.initialize()
        self.learning_method.run()
