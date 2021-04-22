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
Tests the network inference.
"""
import os
import unittest

import tensorflow as tf

from neurallog.knowledge.program import NeuralLogProgram
from neurallog.language.language import Predicate, Constant
from neurallog.language.parser.ply.neural_log_parser import NeuralLogLexer, \
    NeuralLogParser
from neurallog.network.dataset import DefaultDataset
from neurallog.network.network import NeuralLogNetwork

RESOURCES = "network_training"
PROGRAM = "kinship.pl"
EXAMPLES_1 = "kinship_examples_1.pl"
EXAMPLES_2 = "kinship_examples_2.pl"
DATASET_NAME_1 = "examples_1"
DATASET_NAME_2 = "examples_2"

NUMBER_OF_EPOCHS = 10


# noinspection PyMissingOrEmptyDocstring
def get_clauses(filepath):
    # PLY
    lexer = NeuralLogLexer()
    parser = NeuralLogParser(lexer)
    parser.parse(filepath)
    return parser.get_clauses()


# noinspection PyMissingOrEmptyDocstring
class TestNetworkTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        program = get_clauses(os.path.join(RESOURCES, PROGRAM))
        examples_1 = get_clauses(os.path.join(RESOURCES, EXAMPLES_1))
        examples_2 = get_clauses(os.path.join(RESOURCES, EXAMPLES_2))

        # Creates the NeuralLog Program
        cls.program = NeuralLogProgram()  # type: NeuralLogProgram
        cls.program.add_clauses(program)
        cls.program.add_clauses(examples_1, example_set=DATASET_NAME_1)
        cls.program.add_clauses(examples_2, example_set=DATASET_NAME_2)
        cls.program.build_program()

        # Create the dataset
        cls.dataset = DefaultDataset(cls.program, inverse_relations=False)

        # Creates the NeuralLog Model
        cls.model = NeuralLogNetwork(cls.program)
        cls.model.build_layers(cls.dataset.get_target_predicates())
        cls.model.compile(
            loss="mse",
            optimizer=(tf.keras.optimizers.Adagrad(learning_rate=0.1)),
            metrics=["mse"]
        )

    def test_training(self):
        features, _ = self.dataset.build(example_set=DATASET_NAME_1)
        features_2, _2 = self.dataset.build(example_set=DATASET_NAME_2)
        dense_feature_1, _ = self.dataset.call(features, _)
        dense_feature_2, _2 = self.dataset.call(features_2, _2)
        before_training_1 = self.model.predict(dense_feature_1)
        before_training_2 = self.model.predict(dense_feature_2)
        train_set = self.dataset.get_dataset(
            batch_size=1, example_set=DATASET_NAME_1)
        target_index = self.program.get_index_of_constant(
            Predicate("parents", 3), 2, Constant("lucia"))
        value_before = before_training_1[0][target_index]
        for example in self.program.facts_by_predicate.get(
                Predicate("father", 2)).values():
            print(example)
        print("Value before training:", value_before, sep="\t")
        self.model.fit(train_set, epochs=NUMBER_OF_EPOCHS, verbose=0)
        after_training_1 = self.model.predict(dense_feature_1)
        after_training_2 = self.model.predict(dense_feature_2)
        value_after = after_training_1[0][target_index]
        print("Value after training:", value_after, sep="\t")
        self.assertGreater(value_after, value_before)
        # noinspection PyUnresolvedReferences
        self.assertTrue((before_training_2 == after_training_2).all())
        self.model.update_program()
        for example in self.program.facts_by_predicate.get(
                Predicate("father", 2)).values():
            print(example)
            self.assertTrue(
                0.0 <= example.weight <= 1.0,
                "Fact {} outside the constraint range of [0, 1]".format(
                    example.weight))
