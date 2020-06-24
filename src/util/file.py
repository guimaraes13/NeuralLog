"""
Contains useful method to read and write files.
"""
from src.knowledge.examples import ExamplesInferences, ExampleIterator
from src.language.language import AtomClause
from src.language.parser.ply.neural_log_parser import NeuralLogLexer, \
    NeuralLogParser


def read_logic_program_from_file(filepath):
    """
    Reads the logic program from `filepath`.

    :param filepath: the filepath
    :type filepath: str
    :return: the parsed clauses
    :rtype: collections.Collection[Clause]
    """
    # PLY
    lexer = NeuralLogLexer()
    parser = NeuralLogParser(lexer)
    parser.parse(filepath)
    clauses = parser.get_clauses()
    return clauses


def print_predictions_to_file(examples, predictions, filepath):
    """
    Prints the `predictions` to the file in `filepath`.

    :param examples: the examples
    :type examples: Examples
    :param predictions: the predictions
    :type predictions: ExamplesInferences
    :param filepath: the file path
    :type filepath: str
    """
    output = open(filepath, "w")
    for atom in ExampleIterator(examples):
        if predictions.contains_example(atom):
            clause = AtomClause(atom)
            clause.atom.weight = predictions.get_value_for_example(atom)
            output.write(str(clause))
            output.write("\n")
    output.close()
