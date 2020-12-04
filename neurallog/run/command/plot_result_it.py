"""
Plots the results of the iterative structure learning.
"""
import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

from neurallog.knowledge.theory.evaluation.metric.theory_metric import \
    integrate_curve
from neurallog.run.command import command, Command
from neurallog.structure_learning.structure_learning_method import \
    get_sorted_directories
from neurallog.util import print_args
from neurallog.util.statistics import IterationStatistics

COMMAND_NAME = "plot_it"

STATISTICS_FILE_NAME = "statistics.yaml"
RUN_PREFIX = "run_"

logger = logging.getLogger(__name__)


@command(COMMAND_NAME)
class PlotResultIterative(Command):
    """
    Plots the results of the iterative structure learning.
    """

    def __init__(self, program, args, direct=False):
        self.input_directories = None
        self.metric_name = None
        self.output_directory = None
        super().__init__(program, args, direct)

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
        parser.add_argument("--inputDirectories", "-i", metavar="data",
                            type=str, required=True,
                            nargs="*",
                            help="The statistic file(s).")

        parser.add_argument("--metricName", "-m", metavar="metric",
                            type=str, required=True,
                            help="The name of the metric to plot.")

        # Output
        parser.add_argument("--outputDirectory", "-o", metavar="output",
                            type=str, default=None, required=True,
                            help="The file path of the output.")
        return parser

    # noinspection PyMissingOrEmptyDocstring,DuplicatedCode
    def parse_args(self):
        args = self.parser.parse_args(self.args)
        super().parse_args()

        self.input_directories = args.inputDirectories
        self.metric_name = args.metricName
        self.output_directory = args.outputDirectory

        print_args(args, logger)

    # noinspection PyMissingOrEmptyDocstring
    def run(self):
        all_results = dict()
        relation_max_length = dict()

        for input_directory in self.input_directories:
            input_name = os.path.basename(input_directory)
            relation_directories = os.listdir(input_directory)
            for relation in relation_directories:
                relation_directory = os.path.join(input_directory, relation)
                if not os.path.isdir(relation_directory):
                    continue
                run_directories = \
                    get_sorted_directories(relation_directory, RUN_PREFIX)
                if not run_directories:
                    continue
                for run_directory in run_directories:
                    filepath = os.path.join(
                        relation_directory, run_directory, STATISTICS_FILE_NAME)
                    stream = open(filepath, 'r')
                    statistics: IterationStatistics = \
                        yaml.load(stream, Loader=yaml.FullLoader)
                    stream.close()
                    result = [x[self.metric_name] for x in
                              statistics.iteration_test_evaluation]
                    data = all_results.setdefault(relation, dict())
                    data.setdefault(input_name, []).append(result)
                    max_length = relation_max_length.get(relation, 0)
                    relation_length = len(result)
                    if relation_length > max_length:
                        relation_max_length[relation] = relation_length

        for relation, data_results in all_results.items():
            relation_length = relation_max_length[relation]
            relation_space = np.linspace(0.0, 1.0, relation_length)
            relation_range = list(range(relation_length))
            for data, results in data_results.items():
                complete_results = []
                for result in results:
                    if len(result) < relation_length:
                        continue
                    complete_results.append(result)
                np_result = np.array(complete_results).mean(axis=0)
                area = integrate_curve(list(zip(relation_space, np_result)))
                plt.plot(
                    relation_range, np_result,
                    label="{} Final: {:0.4f} Area: {:0.4f}".format(
                        data, np_result[-1], area))
            plt.title(relation)
            plt.xlabel("Iteration")
            plt.ylabel(self.metric_name)
            plt.legend(loc="lower right")
            output_file = os.path.join(self.output_directory, relation + ".pdf")
            plt.savefig(output_file, bbox_inches='tight')
            plt.close()
