"""
Plots the results of the iterative structure learning.
"""
import argparse
import logging

import matplotlib.pyplot as plt
import yaml

from src.knowledge.theory.evaluation.metric.theory_metric import integrate_curve
from src.run.command import command, Command
from src.util import print_args
from src.util.statistics import IterationStatistics
import numpy as np

COMMAND_NAME = "plot_it"

logger = logging.getLogger(__name__)


@command(COMMAND_NAME)
class PlotResultIterative(Command):
    """
    Plots the results of the iterative structure learning.
    """

    def __init__(self, program, args, direct=False):
        self.input_files = None
        self.metric_name = None
        self.output_file = None
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
        parser.add_argument("--inputFiles", "-i", metavar="statistics.yaml",
                            type=str, required=True,
                            nargs="*",
                            help="The statistic file(s).")

        parser.add_argument("--metricName", "-m", metavar="metric",
                            type=str, required=True,
                            help="The name of the metric to plot.")

        # Output
        parser.add_argument("--outputFile", "-o", metavar="output",
                            type=str, default=None, required=True,
                            help="The file path of the output.")
        return parser

    # noinspection PyMissingOrEmptyDocstring,DuplicatedCode
    def parse_args(self):
        args = self.parser.parse_args(self.args)
        super().parse_args()

        self.input_files = args.inputFiles
        self.metric_name = args.metricName
        self.output_file = args.outputFile

        print_args(args, logger)

    # noinspection PyMissingOrEmptyDocstring
    def run(self):
        for filepath in self.input_files:
            stream = open(filepath, 'r')
            statistics: IterationStatistics = \
                yaml.load(stream, Loader=yaml.FullLoader)
            result = [x[self.metric_name] for x in
                      statistics.iteration_test_evaluation]

            length = len(result)
            area = \
                integrate_curve(list(zip(np.linspace(0.0, 1., length), result)))
            plt.plot(range(length), result,
                     label="Final: {:0.4f} Area: {:0.4f}".format(
                         result[-1], area))
            stream.close()

        plt.xlabel("Iteration")
        plt.ylabel(self.metric_name)
        plt.legend(loc="lower right")
        plt.savefig(self.output_file, bbox_inches='tight')
