"""
Tests the structure learning.
"""
import logging
import os
import unittest

from src.language.language import Predicate, HornClause, Atom, Variable, Literal
from src.run import configure_log
from src.run.command.learn_structure import load_yaml_configuration, LOG_FORMAT

PARENT = Predicate("parent", 2)

RESOURCES = "structure_learning"
CONFIGURATION = "configuration.yaml"
PROGRAM = "kinship.pl"
EXAMPLES_1 = "kinship_examples_1.pl"
EXAMPLES_2 = "kinship_examples_2.pl"

CLAUSES = {
    HornClause(
        Atom(PARENT, Variable("A"), Variable("B")),
        Literal(Atom(Predicate("father", 2), Variable("A"), Variable("B")))),
    HornClause(
        Atom(PARENT, Variable("A"), Variable("B")),
        Literal(Atom(Predicate("mother", 2), Variable("A"), Variable("B"))))
}


class TestStructureLearning(unittest.TestCase):

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        configure_log(LOG_FORMAT, level=logging.DEBUG)
        cls.learning_method = \
            load_yaml_configuration(os.path.join(RESOURCES, CONFIGURATION))

    def test_structure_learning(self):
        self.learning_method.initialize()
        self.learning_method.run()
        clauses = self.learning_method.theory.clauses_by_predicate.get(PARENT)
        self.assertIsNotNone(
            clauses, "Expected two clauses in the theory, but it is none.")
        expected = len(CLAUSES)
        found = len(clauses)
        self.assertEqual(
            expected, found,
            "Expected {} clause, but {} found".format(expected, found))
        expected = "\n".join(map(lambda x: str(x), CLAUSES))
        found = "\n".join(map(lambda x: str(x), clauses))
        self.assertTrue(CLAUSES.issuperset(clauses),
                        "Expected:\n{}\n\n{}".format(expected, found))
