"""
Tests the language parser.
"""
import logging
import unittest
from typing import List

from src.language.language import HornClause, Variable, Constant, ListTerms, \
    Quote, Number, TemplateTerm
from src.language.parser.ply.neural_log_parser import NeuralLogLexer, \
    NeuralLogParser
from src.run import configure_log


def _read_program(program):
    """
    Reads the meta program.

    :return: the list of meta clauses
    :rtype: List[HornClause]
    """
    lexer = NeuralLogLexer()
    parser = NeuralLogParser(lexer)
    parser.parser.parse(input=program, lexer=lexer)
    # parser.expand_placeholders()
    # noinspection PyTypeChecker
    return list(parser.get_clauses())


class TestStructureLearning(unittest.TestCase):

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        configure_log(level=logging.DEBUG)

    def test_parser(self):
        program = """
            parent(A, B) :- 
                mother([A, b, ["Cd", 4, test{2}], 3.14, test{3}], E).
            parent(A, B) :- father({test}A, B).
        """

        clauses: List[HornClause] = _read_program(program)
        self.assertEqual(2, len(clauses))
        terms = clauses[0].body[0].terms
        expected = [
            Variable("A"),
            Constant("b"),
            ListTerms((
                Quote("\"Cd\""),
                Number(4),
                TemplateTerm(["test", "{2}"])
            )),
            Number(3.14),
            TemplateTerm(["test", "{3}"])
        ]
        self.assertEqual(ListTerms(expected), terms[0])
        self.assertEqual(Variable("E"), terms[1])

        terms = clauses[1].body[0].terms
        self.assertEqual(TemplateTerm(["{test}", "A"]), terms[0])
