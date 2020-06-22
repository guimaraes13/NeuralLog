"""
Tests the network inference.
"""
import os
import unittest
from typing import List

import numpy as np

from src.knowledge.program import NeuralLogProgram
from src.language.parser.ply.neural_log_parser import NeuralLogLexer
from src.language.parser.ply.neural_log_parser import NeuralLogParser
from src.network.dataset import get_predicate_indices, DefaultDataset
from src.network.network import NeuralLogNetwork

RESOURCES = "network_inference"
PROGRAM = "kinship.pl"
EXAMPLES = "kinship_examples.pl"
EXAMPLES_ARITY_1 = "kinship_examples_arity_1.pl"
EXAMPLES_ARITY_2 = "kinship_examples_arity_2.pl"
EXAMPLES_ARITY_3 = "kinship_examples_arity_3.pl"
DATASET_NAME = "examples"

EQUAL_DELTA = 3

CORRECT = {
    "avgAgeFriends": {
        "andrew": 32.0,
        "christopher": 27.0,
        "james": 18.0,
        "maria": 0.0,
        "sophia": 0.0,
        "charlotte": 0.0
    },
    "similarity": {
        "andrew": {
            "andrew": 1.0,
            "christopher": 0.96592915,
            "james": 0.8660382
        },
        "christopher": {
            "andrew": 0.96592915,
            "christopher": 1.0,
            "james": 0.96592915
        },
        "james": {
            "andrew": 0.8660382,
            "christopher": 0.96592915,
            "james": 1.0
        }
    },
    "similarity^{-1}": {
        "andrew": {
            "andrew": 1.0000001,
            "christopher": 0.96592915,
            "james": 0.8660382
        },
        "christopher": {
            "andrew": 0.96592903,
            "christopher": 1.0,
            "james": 0.96592903
        },
        "james": {
            "andrew": 0.8660382,
            "christopher": 0.96592915,
            "james": 1.0000001
        }
    },
    "parents": {
        "maria": {
            "emilio": 2.0,
            "lucia": 1.0
        }
    },
    "parents^{-1}": {
        "lucia": {
            "sophia": 1.0
        }
    },
    "grand_mother": {
        "maria": {
            "sophia": 1.0
        }
    },
    "grand_mother^{-1}": {
        "sophia": {
            "maria": 1.0
        }
    },
    "grand_grand_father": {
        "andrew": {
            "charlotte": 1.0
        }
    },
    "grand_grand_father^{-1}": {
        "charlotte": {
            "andrew": 1.0
        }
    },
    "wrong_x^{-1}": {
        "andrew": {
            "alfonso": 6.0,
        }
    }
}

CORRECT_SIMILARITY = {
    ("andrew", "andrew", "andrew"): 0.7745132,
    ("andrew", "andrew", "christopher"): 0.7071068,
    ("andrew", "andrew", "james"): 0.59151715,

    ("andrew", "christopher", "christopher"): 0.6830151,
    ("andrew", "christopher", "james"): 0.6123814,

    ("andrew", "james", "james"): 0.59151715,

    ("christopher", "christopher", "christopher"): 0.70710677,
    ("christopher", "christopher", "james"): 0.68301505,

    ("christopher", "james", "james"): 0.70710677,

    ("james", "james", "james"): 0.77451307,
}


# noinspection PyMissingOrEmptyDocstring
def get_clauses(filepath):
    # PLY
    lexer = NeuralLogLexer()
    parser = NeuralLogParser(lexer)
    parser.parse(filepath)
    return parser.get_clauses()


# noinspection DuplicatedCode,PyMissingOrEmptyDocstring
def setup(examples_files=EXAMPLES, inverse_relations=False):
    program = get_clauses(os.path.join(RESOURCES, PROGRAM))
    examples = []
    if not isinstance(examples_files, list):
        examples.append(get_clauses(os.path.join(RESOURCES, examples_files)))
    else:
        for examples_file in examples_files:
            examples.append(get_clauses(os.path.join(RESOURCES, examples_file)))

    # Creates the NeuralLog Program
    neural_program = NeuralLogProgram()  # type: NeuralLogProgram
    neural_program.add_clauses(program)
    for e in examples:
        neural_program.add_clauses(e, example_set=DATASET_NAME)
    neural_program.build_program()

    # Create the dataset
    dataset = DefaultDataset(
        neural_program, inverse_relations=inverse_relations)

    # Creates the NeuralLog Model
    model = NeuralLogNetwork(neural_program)
    model.build_layers(dataset.get_target_predicates())
    return dataset, model


# noinspection PyMissingOrEmptyDocstring
class TestNetworkInference(unittest.TestCase):

    def predict(self, model, features):
        predictions = model.predict(features)  # type: List[np.ndarray]
        predicates = list(model.predicates)
        if len(predicates) == 1:
            # noinspection PyTypeChecker
            if predicates[0][0].arity < 3:
                features = [features]
            predictions = [predictions]
        print("*" * 10, "predictions", "*" * 10)
        offset = 0
        for i in range(len(predicates)):
            x_numpy = []
            summation = 0.0
            for j in range(offset, offset + model.input_sizes[i]):
                array = features[j].numpy()
                x_numpy.append(array)
                summation += array.max()
            offset += model.input_sizes[i]
            if summation == 0.0:
                continue
            prediction = predictions[i]
            predicate = predicates[i][0]
            if predicate.arity > 1:
                for j in range(len(prediction)):
                    indices = np.where(prediction[j] != 0.0)[0]
                    if len(indices) == 0:
                        continue
                    in_indices, output_index = get_predicate_indices(
                        *(predicates[i]))
                    count = 0
                    terms = []
                    stop = False
                    for term_index in in_indices:
                        if x_numpy[count].dtype == np.float32:
                            arg_max = np.argmax(x_numpy[count][j])
                        else:
                            arg_max = x_numpy[count][j][0]
                        count += 1
                        terms.append(model.program.get_constant_by_index(
                            predicates[i][0], term_index, arg_max))
                    if stop:
                        continue
                    terms = list(map(lambda w: str(w), terms))
                    str_terms = ", ".join(terms)

                    name = predicate.name
                    if predicates[i][1]:
                        name += "^{-1}"
                    print(name, "(", str_terms, ", X):", sep="")
                    for index in indices:
                        # if np.isnan(prediction[j][index]):
                        #     continue
                        pred = prediction[j][index]
                        obj = model.program.get_constant_by_index(
                            predicate, output_index, index)
                        print(pred, obj, sep=":\t")
                        if predicate.arity == 2:
                            # noinspection PyUnresolvedReferences
                            expected = CORRECT[name][terms[0]][obj.value]
                        else:
                            key = tuple(sorted(terms + [obj.value]))
                            # noinspection PyTypeChecker
                            expected = CORRECT_SIMILARITY[key]
                        self.assertAlmostEqual(expected, pred, EQUAL_DELTA)
                    print()
            else:
                x_numpy = x_numpy[0]
                name = predicate.name
                if predicates[i][1]:
                    name += "^{-1}"
                print(name, "(X):", sep="")
                for j in range(x_numpy.shape[0]):
                    if x_numpy[j].sum() == 0.0:
                        continue
                    sub = model.program.get_constant_by_index(
                        predicate, 0, np.argmax(x_numpy[j]))
                    pred = prediction[j]
                    print(pred, sub, sep=":\t")
                    expected = CORRECT[name][sub.value]
                    self.assertAlmostEqual(expected, pred, EQUAL_DELTA)
                print()
        print()

    def test_arity_1(self):
        dataset, model = setup(EXAMPLES_ARITY_1)
        self.run_test(dataset, model)

    def test_arity_2(self):
        dataset, model = setup(EXAMPLES_ARITY_2)
        self.run_test(dataset, model)

    def test_arity_3(self):
        dataset, model = setup(EXAMPLES_ARITY_3)
        self.run_test(dataset, model)

    def test_all(self):
        dataset, model = setup(EXAMPLES, inverse_relations=True)
        self.run_test(dataset, model)

    def run_test(self, dataset, model):
        features, _ = dataset.build(example_set=DATASET_NAME)
        dense_feature, _ = dataset.call(features, _)
        self.predict(model, dense_feature)
