"""
Tests the TensorFactory.
"""

import unittest

from antlr4 import FileStream, CommonTokenStream

from src.knowledge.tensor_factory import TensorFactory
from src.language.parser.autogenerated.NeuralLogLexer import NeuralLogLexer
from src.language.parser.autogenerated.NeuralLogParser import NeuralLogParser
from src.language.parser.neural_log_listener import NeuralLogTransverse


class TestTensorFactory(unittest.TestCase):

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        # Create the lexer
        lexer = NeuralLogLexer(FileStream("resources/kinship.pl", "utf-8"))
        # Perform the lexer analysis
        stream = CommonTokenStream(lexer)
        # Create the parser
        parser = NeuralLogParser(stream)
        # Parse the tokens from the input
        abstract_syntax_tree = parser.program()
        # Create the Tree transverse
        transverse = NeuralLogTransverse()
        # Transverse the Abstract Syntax Tree
        neural_program = transverse(abstract_syntax_tree)

        # Create the TensorFactory
        cls.tensor_factory = TensorFactory(neural_program)

    def test_something(self):
        self.assertEqual(True, True)

    def test_something_2(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()