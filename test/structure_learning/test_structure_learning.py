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
Tests the structure learning.
"""

import logging
import os
import unittest
from functools import reduce

from neurallog.language.language import Predicate
from neurallog.language.parser.ply.neural_log_parser import NeuralLogLexer, \
    NeuralLogParser
from neurallog.run import configure_log
from neurallog.run.command.learn_structure import load_yaml_configuration, \
    LOG_FORMAT

PARENT = Predicate("parent", 2)

RESOURCES = "structure_learning"


def _read_program(program):
    """
    Reads the meta program.

    :return: the list of meta clauses
    :rtype: List[HornClause]
    """
    lexer = NeuralLogLexer()
    parser = NeuralLogParser(lexer)
    parser.parser.parse(input=program, lexer=lexer)
    parser.expand_placeholders()
    # noinspection PyTypeChecker
    return set(parser.get_clauses())


class TestStructureLearning(unittest.TestCase):

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        configure_log(LOG_FORMAT, level=logging.DEBUG)

    def _test_structure_learning(
            self, configuration, expected_clauses, assert_equals=True):
        learning_method = \
            load_yaml_configuration(os.path.join(RESOURCES, configuration))
        learning_method.initialize()
        learning_method.run()
        clauses = learning_method.theory.clauses_by_predicate.values()
        clauses = reduce(lambda x, y: x | set(y), clauses, set())
        self.assertIsNotNone(
            clauses, "Expected two clauses in the theory, but it is none.")
        if expected_clauses is not None:
            expected = len(expected_clauses)
            found = len(clauses)
            self.assertEqual(
                expected, found,
                "Expected {} clause, but {} found".format(expected, found))
            expected = "\n".join(map(lambda x: str(x), expected_clauses))
            found = "\n".join(map(lambda x: str(x), clauses))
            if assert_equals:
                self.assertTrue(expected_clauses.issuperset(clauses),
                                "Expected:\n{}\n\n{}".format(expected, found))

    def test_structure_learning(self):
        program = """
            parent(A, B) :- father(A, B).
            parent(A, B) :- mother(A, B).
        """

        self._test_structure_learning(
            "configuration.yaml", _read_program(program))

    def test_tree_structure_learning(self):
        program = """
            parent(X0, X1) :- father(X0, X1).
            parent(X0, X1) :- mother(X0, X1).
        """

        self._test_structure_learning(
            "configuration_tree.yaml", _read_program(program),
            assert_equals=False)

    def test_meta_structure_learning(self):
        program = """
            parent(A, B) :- father(A, B).
            parent(A, B) :- mother(A, B).
        """

        self._test_structure_learning(
            "configuration_meta.yaml", _read_program(program))

    def test_meta_structure_learning_invention(self):
        program = """
            parent(A, B) :- f0(A, B).
            f0(A, B) :- father(A, B).
            f0(A, B) :- mother(A, B).
        """

        self._test_structure_learning(
            "configuration_meta_2.yaml", _read_program(program))

    def test_meta_tree_structure_learning(self):
        program = """
            parent(X0, X1) :- father(X0, X1).
            parent(X0, X1) :- mother(X0, X1).
        """

        self._test_structure_learning(
            "configuration_meta_tree.yaml", _read_program(program))

    def test_tree_structure_learning_it(self):
        self._test_structure_learning("configuration_it.yaml", None)
