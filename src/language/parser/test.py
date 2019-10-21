"""
Testing
"""

import logging
import sys


def configure_log():
    """
    Configures the log handler, message format and log level
    """
    level = logging.DEBUG
    h1 = logging.StreamHandler(sys.stdout)
    h1.setLevel(level)
    h1.addFilter(lambda record: record.levelno <= level)
    h2 = logging.StreamHandler(sys.stderr)
    h2.setLevel(logging.WARNING)
    handlers = [h1, h2]
    # handlers = [h1]
    logging.basicConfig(
        format='%(message)s',
        level=level,
        handlers=handlers
    )


configure_log()

from antlr4 import *

from src.knowledge.network import NeuralLogNetwork, find_clause_paths
from src.knowledge.program import NeuralLogProgram
from src.language.language import Predicate
from src.language.parser.autogenerated.NeuralLogLexer import NeuralLogLexer
from src.language.parser.autogenerated.NeuralLogParser import NeuralLogParser
from src.language.parser.neural_log_listener import NeuralLogTransverse


def main(argv):
    """
    Main function.

    :param argv: the arguments
    :type argv: list[str]
    """
    lexer = NeuralLogLexer(FileStream(argv[1], "utf-8"))
    stream = CommonTokenStream(lexer)
    parser = NeuralLogParser(stream)
    abstract_syntax_tree = parser.program()

    transverse = NeuralLogTransverse()

    print("\n\nNeuralLog Program:\n")
    neural_program = transverse(abstract_syntax_tree)  # type: NeuralLogProgram

    print("Facts: ({})".format(
        sum(map(lambda x: len(x), neural_program.facts_by_predicate.values()))
    ))
    for predicate in sorted(neural_program.facts_by_predicate.keys(),
                            key=lambda x: x.__str__()):
        for fact in neural_program.facts_by_predicate[predicate].values():
            print(fact)

    print("\n\nClauses: ({})".format(
        sum(map(lambda x: len(x), neural_program.clauses_by_predicate.values()))
    ))
    targets = dict()
    for predicate in sorted(neural_program.clauses_by_predicate.keys(),
                            key=lambda x: x.__str__()):
        for clause in neural_program.clauses_by_predicate[predicate]:
            targets[clause.head.predicate.name] = clause
            print(clause)

    print("\n\nPredicates: ({})".format(len(neural_program.predicates)))
    for predicate in sorted(neural_program.predicates.keys(),
                            key=lambda x: x.__str__()):
        if predicate in neural_program.logic_predicates:
            print("LOGIC   ", predicate, sep="\t")
        else:
            print("FUNCTION", predicate, sep="\t")

    print("\n\nTrainables: ({})".format(
        len(neural_program.trainable_predicates)))
    for predicate in sorted(neural_program.trainable_predicates,
                            key=lambda x: x.__str__()):
        print(predicate)

    print("\n\nIterable Constants: ({})"
          .format(len(neural_program.iterable_constants)))
    iterable_constants = set()
    for k, v in neural_program.iterable_constants.items():
        iterable_constants.add(v)
        print("{}:\t{}".format(k, v))

    other_constants = neural_program.constants - iterable_constants
    print("\n\nOther Constants: ({})"
          .format(len(other_constants)))
    for constant in other_constants:
        print(constant)

    print("\n\nExamples: ({})".format(
        sum(map(lambda x: len(x), neural_program.examples.values()))
    ))
    for predicate in sorted(neural_program.examples.keys(),
                            key=lambda x: x.__str__()):
        for fact in neural_program.examples[predicate].values():
            print(fact)

    father_matrix = neural_program.get_matrix_representation(Predicate(
        "father", 2))
    mother_matrix = neural_program.get_matrix_representation(Predicate(
        "mother", 2))

    # print("\nFather Matrix:")
    # print(father_matrix)
    # print("\nMother Matrix:")
    # print(mother_matrix)

    grand_father = father_matrix.dot(father_matrix) + \
                   father_matrix.dot(mother_matrix)

    print()
    for i, j in zip(*grand_father.nonzero()):
        print("{}::{}({}, {}).".format(grand_father[i, j],
                                       "grand_father",
                                       neural_program.iterable_constants[i],
                                       neural_program.iterable_constants[j]))

    print("\n")
    test = Predicate("test")
    print("{}::{}.".format(neural_program.get_matrix_representation(test),
                           test.name))

    print("\n")
    male = Predicate("male", 1)
    print(male)
    representation = neural_program.get_matrix_representation(male)
    for i, j in zip(*representation.nonzero()):
        print("{}::{}({}).".format(
            representation[i, j],
            "male",
            neural_program.iterable_constants[i]
        ))

    print("\n")
    age = Predicate("age", 2)
    print(age)
    age_weights, age_attributes = neural_program.get_matrix_representation(age)
    for i, j in zip(*age_weights.nonzero()):
        print("{}::{}({}, {}).".format(
            age_weights[i, j],
            "age",
            neural_program.iterable_constants[i],
            age_attributes[i, j],
        ))

    network = NeuralLogNetwork(neural_program)
    # network.build_network()
    paths, grounded_literals = find_clause_paths(targets["target"])
    string_literals = ", ".join(map(lambda x: x.__str__(), grounded_literals))
    rev_string_literals = ", ".join(
        reversed(list(map(lambda x: x.__str__(), grounded_literals))))

    print("Paths:")
    for path in paths:
        print("Path:\t{}, {}\t<->\t{}, {}".format(
            path, string_literals, rev_string_literals, path.reverse()))


if __name__ == "__main__":
    configure_log()
    main(sys.argv)
