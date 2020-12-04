"""
Contains useful method to read and write files.
"""
import sys

from neurallog.knowledge.examples import ExamplesInferences, ExampleIterator
from neurallog.language.language import AtomClause, Atom
from neurallog.language.parser.ply.neural_log_parser import NeuralLogLexer, \
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


def print_predictions_to_file(examples, predictions, filepath,
                              default_value=-sys.float_info.max):
    """
    Prints the `predictions` to the file in `filepath`.

    :param examples: the examples
    :type examples: Examples
    :param predictions: the predictions
    :type predictions: ExamplesInferences
    :param filepath: the file path
    :type filepath: str
    :param default_value: the default value to save not inferred examples
    :type default_value: float
    """
    output = open(filepath, "w")
    for atom in ExampleIterator(examples):
        clause = AtomClause(atom)
        if predictions.contains_example(atom):
            clause.atom.weight = predictions.get_value_for_example(atom)
        else:
            clause.atom.weight = default_value
        output.write(str(clause))
        output.write("\n")
    output.close()


TSV_HEADER = "Example\tExpected\tInference\n"


def print_tsv_file_file(examples, predictions, filepath,
                        default_value=-sys.float_info.max):
    """
    Prints the `predictions` to the file in `filepath`, in tsv format.

    :param examples: the examples
    :type examples: Examples
    :param predictions: the predictions
    :type predictions: ExamplesInferences
    :param filepath: the file path
    :type filepath: str
    :param default_value: the default value to save not inferred examples
    :type default_value: float
    """
    output = open(filepath, "w")
    output.write(TSV_HEADER)
    for atom in ExampleIterator(examples):
        clause = Atom(atom.predicate, *atom.terms)
        if predictions.contains_example(atom):
            pred = predictions.get_value_for_example(atom)
        else:
            pred = default_value
        output.write(f"{clause}\t{atom.weight}\t{pred}\n")
    output.close()
