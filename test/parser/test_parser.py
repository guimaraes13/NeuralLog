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
Tests the language parser.
"""
import logging
import unittest
from typing import List

from neurallog.knowledge.program import NeuralLogProgram
from neurallog.language.language import HornClause, Variable, Constant, \
    ListTerms, Quote, Number, TemplateTerm, Predicate
from neurallog.language.parser.ply.neural_log_parser import NeuralLogLexer, \
    NeuralLogParser
from neurallog.run import configure_log


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


class TestParser(unittest.TestCase):

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

    def test_parameters(self):
        parameter_name = "parameter_name"
        predicate = Predicate("predicate_name", 2)
        program = f"""
            set_parameter({parameter_name}, [1, C, 2]).
            set_predicate_parameter("{predicate}", class_name, test).
            set_predicate_parameter("{predicate}", config, [1, [4, 2.2], 3.8]).
        """
        clauses: List[HornClause] = _read_program(program)
        neural_program = NeuralLogProgram()
        neural_program.add_clauses(clauses)
        neural_program.build_program()
        # Predicate
        expect1 = [1, "C", 2]
        expect2 = [1, [4, 2.2], 3.8]

        self.assertTrue(parameter_name in neural_program.parameters)
        values1 = neural_program.parameters[parameter_name]
        self.assertEqual(expect1, values1)
        self.assertTrue(predicate in neural_program.parameters)
        values2 = neural_program.parameters[predicate]["class_name"]
        self.assertEqual("test", values2)
        values3 = neural_program.parameters[predicate]["config"]
        self.assertEqual(expect2, values3)
