"""
Tests the structure learning.
"""
import os
import unittest
import logging

from src.run import configure_log
from src.run.command.learn_structure import load_yaml_configuration, LOG_FORMAT

RESOURCES = "structure_learning"
CONFIGURATION = "configuration.yaml"
PROGRAM = "kinship.pl"
EXAMPLES_1 = "kinship_examples_1.pl"
EXAMPLES_2 = "kinship_examples_2.pl"


class TestStructureLearning(unittest.TestCase):

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        configure_log(LOG_FORMAT, level=logging.DEBUG)
        cls.learning_method = \
            load_yaml_configuration(os.path.join(RESOURCES, CONFIGURATION))

    def test_structure_learning(self):
        self.learning_method.initialize()
        # print(self.learning_method.pass_all_examples_at_once)
        # print(self.learning_method.train_parameters_on_remaining_examples)
        # self.learning_method.run()
