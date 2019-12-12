"""
Tests the network inference.
"""
import unittest
from typing import List

import numpy as np
import tensorflow as tf

from src.knowledge.program import NeuralLogProgram
from src.language.parser.ply.neural_log_parser import NeuralLogLexer
from src.language.parser.ply.neural_log_parser import NeuralLogParser
from src.network.network import NeuralLogNetwork, NeuralLogDataset

RESOURCES = "network_inference"
PROGRAM = "kinship.pl"
EXAMPLES = "kinship_examples.pl"
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
            "andrew": 1.0,
            "christopher": 0.96586794,
            "james": 0.8659618
        },
        "christopher": {
            "andrew": 0.96586794,
            "christopher": 1.0,
            "james": 0.96586794
        },
        "james": {
            "andrew": 0.8659618,
            "christopher": 0.96586794,
            "james": 1.0
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
    }
}


# noinspection PyMissingOrEmptyDocstring
def get_clauses(filepath):
    # PLY
    lexer = NeuralLogLexer()
    parser = NeuralLogParser(lexer)
    parser.parse(filepath)
    return parser.get_clauses()


# noinspection DuplicatedCode,PyMissingOrEmptyDocstring
class TestNetworkInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # program = get_clauses(os.path.join(RESOURCES, PROGRAM))
        # examples = get_clauses(os.path.join(RESOURCES, EXAMPLES))

        program = get_clauses(PROGRAM)
        examples = get_clauses(EXAMPLES)

        # Creates the NeuralLog Program
        cls.program = NeuralLogProgram()  # type: NeuralLogProgram
        cls.program.add_clauses(program)
        cls.program.add_clauses(examples, example_set=DATASET_NAME)
        cls.program.build_program()

        # Creates the NeuralLog Model
        cls.model = NeuralLogNetwork(cls.program)
        cls.model.build_layers()
        # cls.model.compile()

        # Create the dataset
        cls.dataset = NeuralLogDataset(cls.model)

    def predict(self, features):
        predictions = self.model.predict(features)  # type: List[np.ndarray]
        predicates = list(self.model.predicates)
        if len(predicates) == 1:
            # noinspection PyTypeChecker
            predictions = [predictions]
        x_numpy = features.numpy()
        print("*" * 10, "predictions", "*" * 10)
        for i in range(len(predicates)):
            prediction = predictions[i]
            if predicates[i][0].arity == 2:
                for j in range(len(prediction)):
                    indices = np.where(prediction[j] != 0.0)[0]
                    if len(indices) == 0:
                        continue
                    sub = self.program.iterable_constants[
                        np.argmax(x_numpy[j])]
                    name = predicates[i][0].name
                    if predicates[i][1]:
                        name += "^{-1}"
                    print(name, "(", sub, ", X):", sep="")
                    for index in indices:
                        # if np.isnan(prediction[j][index]):
                        #     continue
                        pred = prediction[j][index]
                        obj = self.program.iterable_constants[index]
                        print(pred, obj, sep=":\t")
                        expected = CORRECT[name][sub.value][obj.value]
                        self.assertAlmostEqual(expected, pred, EQUAL_DELTA)
                    print()
            else:
                name = predicates[i][0].name
                if predicates[i][1]:
                    name += "^{-1}"
                print(name, "(X):", sep="")
                for j in range(x_numpy.shape[0]):
                    sub = self.program.iterable_constants[
                        np.argmax(x_numpy[j])]
                    pred = prediction[j]
                    # print(name, "(", sub, "):\t", pred, sep="")
                    print(pred, sub, sep=":\t")
                    expected = CORRECT[name][sub.value]
                    self.assertAlmostEqual(expected, pred, EQUAL_DELTA)
                print()
        print()

    def test_inference(self):
        features, _ = self.dataset.build(example_set=DATASET_NAME)
        dense_feature = tf.one_hot(features, self.model.constant_size)
        self.predict(dense_feature)
