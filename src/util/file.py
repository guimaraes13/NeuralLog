"""
Contains useful method to read and write files.
"""

from src.language.parser.ply.neural_log_parser import NeuralLogLexer, \
    NeuralLogParser


def read_logic_program_from_file(filepath):
    """
    Reads the logic program from `filepath`.

    :param filepath: the filepath
    :type filepath: str
    :return: the parsed clauses
    :rtype: collections.Iterable[Clause]
    """
    # PLY
    lexer = NeuralLogLexer()
    parser = NeuralLogParser(lexer)
    parser.parse(filepath)
    clauses = parser.get_clauses()
    return clauses
