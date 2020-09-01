"""
Tests the structure learning.
"""
import logging
import os
import unittest
from functools import reduce

from src.language.language import Predicate
from src.language.parser.ply.neural_log_parser import NeuralLogLexer, \
    NeuralLogParser
from src.run import configure_log
from src.run.command.learn_structure import load_yaml_configuration, LOG_FORMAT

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
            parent(X0, X1) :- father(X0, X1).
            parent(X0, X1) :- mother(X0, X1).
        """

        self._test_structure_learning(
            "configuration_meta.yaml", _read_program(program))

    def test_meta_structure_learning_invention(self):
        program = """
            parent(X0, X1) :- f0(X0, X1).
            f0(X0, X1) :- father(X0, X1).
            f0(X0, X1) :- mother(X0, X1).
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
