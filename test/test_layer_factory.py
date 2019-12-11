"""
Tests the TensorFactory.
"""

import unittest

import numpy as np
import tensorflow as tf
from antlr4 import FileStream, CommonTokenStream

from src.knowledge.program import NeuralLogProgram
from src.language.language import Atom
from src.language.parser.ply.neural_log_parser import NeuralLogLexer
from src.language.parser.ply.neural_log_parser import NeuralLogParser
from src.network.layer_factory import LayerFactory

NEUTRAL_ELEMENT = tf.constant(1.0)

WEIGHT_MIN_VALUE = 0.02
WEIGHT_MAX_VALUE = 0.05

EQUAL_DELTA = 3


# noinspection DuplicatedCode
class TestLayerFactory(unittest.TestCase):

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        # Create the lexer
        # lexer = NeuralLogLexer(FileStream("layer_factory.pl", "utf-8"))
        # # Perform the lexer analysis
        # stream = CommonTokenStream(lexer)
        # # Create the parser
        # parser = NeuralLogParser(stream)
        # # Parse the tokens from the input
        # abstract_syntax_tree = parser.program()
        # # Create the Tree transverse
        # transverse = NeuralLogTransverse()
        # # Transverse the Abstract Syntax Tree
        # clauses = transverse(abstract_syntax_tree)

        lexer = NeuralLogLexer()
        parser = NeuralLogParser(lexer)
        parser.parse("layer_factory.pl")
        clauses = parser.clauses[0]
        # Create the NeuralLog Program
        cls.program = NeuralLogProgram()
        cls.program.add_clauses(clauses)
        cls.program.build_program()

        # Create the TensorFactory
        cls.layer_factory = LayerFactory(cls.program)

    def test_arity_0_not_trainable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        atom = Atom("predicate_0_not_trainable", weight=0.3)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_0_not_trainable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)
        evaluated = tensor.numpy()

        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated, places=EQUAL_DELTA)

    def test_arity_0_trainable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        atom = Atom("predicate_0_trainable", weight=0.5)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_0_trainable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated, places=EQUAL_DELTA)

    def test_arity_1_0_not_trainable_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 2019
        atom = Atom("year", value, weight=0.7)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_1_0_not_trainable_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * value, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_1_0_not_trainable_number_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 2021
        atom = Atom("year", value, weight=0.7)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_1_0_not_trainable_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_1_0_not_trainable_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        expected = 2019
        atom = Atom("year", value, weight=0.7)
        # atom = Atom("multiply", "X", "Y", weight=0.5)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_1_0_not_trainable_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * expected, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_1_0_trainable_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 21
        atom = Atom("century", value, weight=0.9)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_1_0_trainable_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * value, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_1_0_trainable_number_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 22
        atom = Atom("century", value)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_1_0_trainable_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertLessEqual(evaluated, WEIGHT_MAX_VALUE * (value + 1))
        self.assertGreaterEqual(evaluated, WEIGHT_MIN_VALUE * (value - 1))

    def test_arity_1_0_trainable_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        expected = 21
        atom = Atom("century", value, weight=0.9)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_1_0_trainable_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * expected, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_1_1_not_trainable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "some_male"
        atom = Atom("male", value, weight=0.07)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_1_1_not_trainable_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated, places=EQUAL_DELTA)

    def test_arity_1_1_not_trainable_constant_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "some_male2"
        atom = Atom("male", value, weight=0.0)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_1_1_not_trainable_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated, places=EQUAL_DELTA)

    def test_arity_1_1_not_trainable_iterable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "christopher"
        atom = Atom("male", value, weight=0.07)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_1_1_not_trainable_iterable_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated, places=EQUAL_DELTA)

    def test_arity_1_1_not_trainable_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        atom = Atom("male", value)
        correct = np.array([0.02, 0.03, 0.0, 0.05, 0.0, 0.0, 0.07, 0.11, 0.13,
                            0.0, 0.17, 0.0, 0.0, 0.19, 0.0, 0.0, 0.23, 0.31,
                            0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_1_1_not_trainable_variable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                correct[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_1_1_trainable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "some_female"
        atom = Atom("female", value, weight=0.107)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_1_1_trainable_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated, places=EQUAL_DELTA)

    def test_arity_1_1_trainable_constant_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "some_female2"
        atom = Atom("female", value, weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_1_1_trainable_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertLessEqual(evaluated, atom.weight)
        self.assertGreaterEqual(evaluated, WEIGHT_MIN_VALUE)

    def test_arity_1_1_trainable_iterable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "christine"
        atom = Atom("female", value, weight=0.107)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_1_1_trainable_iterable_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((1, 1), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated, places=EQUAL_DELTA)

    def test_arity_1_1_trainable_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        atom = Atom("female", value)
        weight = np.array(
            [0.0, 0.0, 0.101, 0.0, 0.103, 0.107, 0.0, 0.0, 0.0, 0.109, 0.0,
             0.127, 0.137, 0.0, 0.149, 0.151, 0.0, 0.0, 0.157, 0.163])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_1_1_trainable_variable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            error_message = "Incorrect value for constant {}: {}".format(
                i, layer_factory.program.iterable_constants[i])
            if np.isclose(weight[i], 0.0):
                self.assertGreaterEqual(evaluated[i], WEIGHT_MIN_VALUE,
                                        msg=error_message)
                self.assertLessEqual(evaluated[i], WEIGHT_MAX_VALUE,
                                     msg=error_message)
            else:
                self.assertAlmostEqual(
                    weight[i], evaluated[i], places=EQUAL_DELTA,
                    msg=error_message)

    def test_arity_2_0_not_trainable_number_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = 2.0
        value2 = 3.0
        atom = Atom("multiply", value1, value2, weight=0.5)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_not_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * value1 * value2, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_0_not_trainable_number_number_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = 2.0
        value2 = 5.0
        atom = Atom("multiply", value1, value2, weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_not_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_2_0_not_trainable_number_not_in_kb_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = 5.0
        value2 = 3.0
        atom = Atom("multiply", value1, value2, weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_not_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_2_0_not_trainable_number_not_in_kb_number_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = 5.0
        value2 = 4.0
        atom = Atom("multiply", value1, value2, weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_not_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_2_0_not_trainable_number_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = 2.0
        value2 = "X"
        expected2 = 3.0
        atom = Atom("multiply", value1, value2, weight=0.5)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_not_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * value1 * expected2, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_0_not_trainable_number_variable_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = "X"
        expected1 = 2.0
        value2 = 3.0
        atom = Atom("multiply", value1, value2, weight=0.5)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_not_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * expected1 * value2, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_0_not_trainable_number_variable_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = "X"
        expected1 = 2.0
        value2 = "Y"
        expected2 = 3.0
        atom = Atom("multiply", value1, value2, weight=0.5)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_not_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * expected1 * expected2, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_0_not_trainable_same_number_variables(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = "X"
        value2 = "X"
        atom = Atom("multiply", value1, value2, weight=0.5)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_not_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_2_0_trainable_number_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = 4.0
        value2 = 3.0
        atom = Atom("multiply_2", value1, value2, weight=0.25)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * value1 * value2, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_0_trainable_number_number_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = 4.0
        value2 = 3.4
        atom = Atom("multiply_2", value1, value2, weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertLessEqual(evaluated, atom.weight * value1 * value2)
        self.assertGreaterEqual(evaluated, WEIGHT_MIN_VALUE)

    def test_arity_2_0_trainable_number_not_in_kb_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = 4.3
        value2 = 3.0
        atom = Atom("multiply_2", value1, value2, weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertLessEqual(evaluated, atom.weight * value1 * value2)
        self.assertGreaterEqual(evaluated, WEIGHT_MIN_VALUE)

    def test_arity_2_0_trainable_number_not_in_kb_number_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = 4.3
        value2 = 3.4
        atom = Atom("multiply_2", value1, value2, weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertLessEqual(evaluated, atom.weight * value1 * value2)
        self.assertGreaterEqual(evaluated, WEIGHT_MIN_VALUE)

    def test_arity_2_0_trainable_number_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = 4.0
        value2 = "X"
        expected2 = 3.0
        atom = Atom("multiply_2", value1, value2, weight=0.25)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * value1 * expected2, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_0_trainable_number_variable_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = "X"
        expected1 = 4.0
        value2 = 3.0
        atom = Atom("multiply_2", value1, value2, weight=0.25)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * expected1 * value2, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_0_trainable_number_variable_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = "X"
        expected1 = 4.0
        value2 = "Y"
        expected2 = 3.0
        atom = Atom("multiply_2", value1, value2, weight=0.25)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * expected1 * expected2, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_0_trainable_same_number_variables(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value1 = "X"
        value2 = "X"
        atom = Atom("multiply_2", value1, value2)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_0_trainable_number_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_constant_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 30
        atom = Atom("age", "some_male", value, weight=0.211)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_constant_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * value, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_constant_number_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 35
        atom = Atom("age", "some_male", value, weight=0.211)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_constant_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_constant_not_in_kb_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 30
        atom = Atom("age", "some_male2", value, weight=0.211)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_constant_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_constant_not_in_kb_number_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 35
        atom = Atom("age", "some_male2", value, weight=0.211)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_constant_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_constant_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        expected = 30
        atom = Atom("age", "some_male", value, weight=0.211)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_constant_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * expected, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_iterable_constant_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 41
        atom = Atom("age", "colin", value, weight=0.223)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.
                arity_2_1_not_trainable_iterable_constant_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((1, 1), evaluated.shape)
        self.assertAlmostEqual(atom.weight * value, evaluated[0, 0],
                               places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_iterable_constant_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        expected = 41
        atom = Atom("age", "colin", value, weight=0.223)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.
                arity_2_1_not_trainable_iterable_constant_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((1, 1), evaluated.shape)
        self.assertAlmostEqual(atom.weight * expected, evaluated[0, 0],
                               places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_variable_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 27
        atom = Atom("age", "X", value)
        weight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.227,
                           0.241, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0])
        values = np.array([0, 0, 0, 0, 0, 0, 0, 41, 27, 27, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0])
        correct = weight * values
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_variable_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                correct[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_1_not_trainable_variable_number_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 105
        atom = Atom("age", "X", value)
        weight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_variable_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                weight[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_1_not_trainable_variable_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "Y"
        atom = Atom("age", "X", value)
        weight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.223, 0.227,
                           0.241, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0])
        values = np.array([0, 0, 0, 0, 0, 0, 0, 41, 27, 27, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0])
        correct = weight * values
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_variable_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                correct[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_1_not_trainable_equal_variable_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        atom = Atom("age", "X", value)
        weight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_variable_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                weight[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_1_not_trainable_number_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 1.72
        atom = Atom("inv_height", value, "some_female", weight=0.337)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_number_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * value, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_number_constant_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 1.72
        atom = Atom("inv_height", value, "some_female2", weight=0.337)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_number_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_number_not_in_kb_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 1.75
        atom = Atom("inv_height", value, "some_female", weight=0.337)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_number_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_number_not_in_kb_constant_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 1.75
        atom = Atom("inv_height", value, "some_female2", weight=0.337)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_number_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_number_variable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        expected = 1.72
        atom = Atom("inv_height", value, "some_female", weight=0.337)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_number_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * expected, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_number_iterable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 1.73
        atom = Atom("inv_height", value, "jennifer", weight=0.353)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.
                arity_2_1_not_trainable_number_iterable_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((1, 1), evaluated.shape)
        self.assertAlmostEqual(atom.weight * value, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_number_variable_iterable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        expected = 1.73
        atom = Atom("inv_height", value, "jennifer", weight=0.353)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.
                arity_2_1_not_trainable_number_iterable_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((1, 1), evaluated.shape)
        self.assertAlmostEqual(atom.weight * expected, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_1_not_trainable_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 1.65
        atom = Atom("inv_height", value, "X")
        weight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.359, 0.331, 0.0, 0.0, 0.367, 0.0,
                           0.0, 0.0])
        values = np.array([0.0, 0.0, 0.0, 0.0, 1.81, 0.0, 0.0, 1.57, 2.06,
                           1.7, 1.9, 1.73, 1.65, 1.65, 0.0, 0.0, 1.65, 0.0,
                           0.0, 0.0])
        correct = weight * values
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_number_variable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                correct[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_1_not_trainable_number_not_in_kb_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 2.65
        atom = Atom("inv_height", value, "X")
        weight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_number_variable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                weight[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_1_not_trainable_number_variable_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "Y"
        atom = Atom("inv_height", value, "X")
        weight = np.array([0.0, 0.0, 0.0, 0.0, 0.373, 0.0, 0.0, 0.311, 0.313,
                           0.347, 0.317, 0.353, 0.359, 0.331, 0.0, 0.0,
                           0.367, 0.0, 0.0, 0.0])
        values = np.array([0.0, 0.0, 0.0, 0.0, 1.81, 0.0, 0.0, 1.57, 2.06,
                           1.7, 1.7, 1.73, 1.65, 1.65, 0.0, 0.0, 1.65, 0.0,
                           0.0, 0.0])
        correct = weight * values
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_number_variable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                correct[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_1_not_trainable_equal_number_variable_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        atom = Atom("inv_height", value, "X")
        weight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_not_trainable_number_variable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                weight[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_1_trainable_constant_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 1.73
        atom = Atom("height", "some_male", value, weight=0.307)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_constant_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * value, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_1_trainable_constant_number_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 1.75
        atom = Atom("height", "some_male", value, weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_constant_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertLessEqual(evaluated, atom.weight * value)
        self.assertGreaterEqual(evaluated, WEIGHT_MIN_VALUE * value)

    def test_arity_2_1_trainable_constant_not_in_kb_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 1.73
        atom = Atom("height", "some_male2", value, weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_constant_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertLessEqual(evaluated, atom.weight * value)
        self.assertGreaterEqual(evaluated, WEIGHT_MIN_VALUE * value)

    def test_arity_2_1_trainable_constant_not_in_kb_number_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 1.75
        atom = Atom("height", "some_male2", value, weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_constant_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertLessEqual(evaluated, atom.weight * value)
        self.assertGreaterEqual(evaluated, WEIGHT_MIN_VALUE * value)

    def test_arity_2_1_trainable_constant_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        expected = 1.73
        atom = Atom("height", "some_male", value, weight=0.307)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_constant_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * expected, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_1_trainable_iterable_constant_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 1.7
        atom = Atom("height", "james", value, weight=0.317)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.
                arity_2_1_trainable_iterable_constant_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((1, 1), evaluated.shape)
        self.assertAlmostEqual(atom.weight * value, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_1_trainable_iterable_constant_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        expected = 1.7
        atom = Atom("height", "james", value, weight=0.317)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.
                arity_2_1_trainable_iterable_constant_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((1, 1), evaluated.shape)
        self.assertAlmostEqual(atom.weight * expected, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_1_trainable_variable_number(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 1.7
        atom = Atom("height", "X", value)
        weight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.347, 0.317, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_variable_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            error_message = "Incorrect value for constant {}: {}".format(
                i, layer_factory.program.iterable_constants[i])
            if np.isclose(weight[i], 0.0):
                self.assertGreaterEqual(evaluated[i],
                                        value * WEIGHT_MIN_VALUE,
                                        msg=error_message)
                self.assertLessEqual(evaluated[i], value * WEIGHT_MAX_VALUE,
                                     msg=error_message)
            else:
                self.assertAlmostEqual(
                    weight[i] * value, evaluated[i], places=EQUAL_DELTA,
                    msg=error_message)

    def test_arity_2_1_trainable_variable_number_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 2.7
        atom = Atom("height", "X", value)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_variable_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertGreaterEqual(evaluated[i], value * WEIGHT_MIN_VALUE)
            self.assertLessEqual(evaluated[i], value * WEIGHT_MAX_VALUE)

    def test_arity_2_1_trainable_variable_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "Y"
        atom = Atom("height", "X", value)
        weight = np.array([0.0, 0.0, 0.0, 0.0, 0.373, 0.0, 0.0, 0.311, 0.313,
                           0.347, 0.317, 0.353, 0.359, 0.331, 0.0, 0.0,
                           0.367, 0.0, 0.0, 0.0])
        values = np.array([0.0, 0.0, 0.0, 0.0, 1.81, 0.0, 0.0, 1.57, 2.06,
                           1.7, 1.7, 1.73, 1.59, 1.65, 0.0, 0.0, 1.82, 0.0,
                           0.0, 0.0])
        correct = weight * values
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_variable_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                correct[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_1_trainable_equal_variable_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        atom = Atom("height", "X", value)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_variable_number.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                0.0, evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_1_trainable_number_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 30
        atom = Atom("inv_age", value, "some_male", weight=0.211)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_number_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * value, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_1_trainable_number_constant_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 30
        atom = Atom("inv_age", value, "some_male2", weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_number_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertLessEqual(evaluated, atom.weight * value)
        self.assertGreaterEqual(evaluated, WEIGHT_MIN_VALUE * value)

    def test_arity_2_1_trainable_number_not_in_kb_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 32
        atom = Atom("inv_age", value, "some_male", weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_number_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertLessEqual(evaluated, atom.weight * value)
        self.assertGreaterEqual(evaluated, WEIGHT_MIN_VALUE * value)

    def test_arity_2_1_trainable_number_not_in_kb_constant_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 32
        atom = Atom("inv_age", value, "some_male2", weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_number_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertLessEqual(evaluated, atom.weight * value)
        self.assertGreaterEqual(evaluated, WEIGHT_MIN_VALUE * value)

    def test_arity_2_1_trainable_number_variable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        expected = 30
        atom = Atom("inv_age", value, "some_male", weight=0.211)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_number_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight * expected, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_1_trainable_number_iterable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 27
        atom = Atom("inv_age", value, "francesca", weight=0.241)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.
                arity_2_1_trainable_number_iterable_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((1, 1), evaluated.shape)
        self.assertAlmostEqual(atom.weight * value, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_1_trainable_number_variable_iterable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        expected = 27
        atom = Atom("inv_age", value, "francesca", weight=0.241)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.
                arity_2_1_trainable_number_iterable_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((1, 1), evaluated.shape)
        self.assertAlmostEqual(atom.weight * expected, evaluated,
                               places=EQUAL_DELTA)

    def test_arity_2_1_trainable_number_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 27
        atom = Atom("inv_age", value, "X")
        weight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.227,
                           0.241, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_number_variable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            error_message = "Incorrect value for constant {}: {}".format(
                i, layer_factory.program.iterable_constants[i])
            if np.isclose(weight[i], 0.0):
                self.assertGreaterEqual(evaluated[i],
                                        value * WEIGHT_MIN_VALUE,
                                        msg=error_message)
                self.assertLessEqual(evaluated[i], value * WEIGHT_MAX_VALUE,
                                     msg=error_message)
            else:
                self.assertAlmostEqual(
                    weight[i] * value, evaluated[i], places=EQUAL_DELTA,
                    msg=error_message)

    def test_arity_2_1_trainable_number_not_in_kb_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = 127
        atom = Atom("inv_age", value, "X")
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_number_variable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            error_message = "Incorrect value for constant {}: {}".format(
                i, layer_factory.program.iterable_constants[i])
            self.assertGreaterEqual(evaluated[i],
                                    value * WEIGHT_MIN_VALUE,
                                    msg=error_message)
            self.assertLessEqual(evaluated[i], value * WEIGHT_MAX_VALUE,
                                 msg=error_message)

    def test_arity_2_1_trainable_number_variable_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "Y"
        atom = Atom("inv_age", value, "X")
        weight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.223,
                           0.227, 0.241, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0])
        values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 41,
                           27, 27, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0])
        correct = weight * values
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_number_variable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                correct[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_1_trainable_equal_number_variable_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = "X"
        atom = Atom("inv_age", value, "X")
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_1_trainable_number_variable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                0.0, evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_2_not_trainable_constant_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["some_male", "some_female"]
        atom = Atom("husband", *value, weight=0.6012)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_not_trainable_constant_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated, places=EQUAL_DELTA)

    def test_arity_2_2_not_trainable_constant_constant_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["some_male", "some_female2"]
        atom = Atom("husband", *value, weight=0.6012)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_not_trainable_constant_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_2_2_not_trainable_constant_not_in_kb_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["some_male2", "some_female"]
        atom = Atom("husband", *value, weight=0.6012)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_not_trainable_constant_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_2_2_not_trainable_constant_not_n_kb_constant_not_n_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["some_male2", "some_female2"]
        atom = Atom("husband", *value, weight=0.6012)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_not_trainable_constant_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(0.0, evaluated, places=EQUAL_DELTA)

    def test_arity_2_2_not_trainable_constant_iterable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["some_female", "jennifer"]
        atom = Atom("daughter", *value, weight=0.9415)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_not_trainable_constant_iterable_constant.
                __func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated, places=EQUAL_DELTA)

    def test_arity_2_2_not_trainable_constant_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["some_female", "X"]
        atom = Atom("sister", *value)
        correct = np.array([0.0, 0.0, 0.6853, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.5853, 0.0, 0.7853, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_not_trainable_constant_variable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                correct[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_2_not_trainable_constant_not_in_kb_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["some_female2", "X"]
        atom = Atom("sister", *value)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_not_trainable_constant_variable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                0.0, evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_2_not_trainable_iterable_constant_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["james", "some_female"]
        atom = Atom("husband", *value, weight=0.7019)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_not_trainable_iterable_constant_constant.
                __func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated, places=EQUAL_DELTA)

    def test_arity_2_2_not_trainable_iterable_constant_iterable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["james", "victoria"]
        atom = Atom("husband", *value, weight=0.577)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.
                arity_2_2_not_trainable_iterable_constant_iterable_constant.
                __func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated, places=EQUAL_DELTA)

    def test_arity_2_2_not_trainable_iterable_constant_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["james", "X"]
        atom = Atom("father", *value)
        correct = np.array([0.0, 0.0, 0.0, 0.0, 0.733, 0.0, 0.0, 0.739, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_not_trainable_iterable_constant_variable.
                __func__,
            layer_factory.function[key])
        evaluated = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT).numpy()

        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                correct[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_2_not_trainable_variable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["X", "some_female"]
        atom = Atom("brother", *value)
        correct = np.array([0.0, 0.0, 0.6853, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.5853, 0.0, 0.7853, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_not_trainable_variable_constant.__func__,
            layer_factory.function[key])

        input_value = tf.eye(layer_factory.constant_size)
        evaluated = layer_factory.build_atom(atom)(input_value).numpy()

        self.assertEqual((layer_factory.constant_size, 1), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                correct[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_2_not_trainable_variable_constant_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["X", "some_female2"]
        atom = Atom("brother", *value)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_not_trainable_variable_constant.__func__,
            layer_factory.function[key])
        input_value = tf.eye(layer_factory.constant_size)
        evaluated = layer_factory.build_atom(atom)(input_value).numpy()

        self.assertEqual((layer_factory.constant_size, 1), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                0.0, evaluated[i][0], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_2_not_trainable_variable_iterable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["X", "james"]
        atom = Atom("daughter", *value)
        correct = np.array([0.0, 0.0, 0.0, 0.0, 0.947, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.094, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_not_trainable_variable_iterable_constant.
                __func__,
            layer_factory.function[key])
        input_value = tf.eye(layer_factory.constant_size)
        evaluated = layer_factory.build_atom(atom)(input_value).numpy()

        self.assertEqual((layer_factory.constant_size, 1), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                correct[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constant {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_2_not_trainable_variable_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["X", "Y"]
        atom = Atom("father", *value)
        correct = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.701, 0.709,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.719, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.727],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.733, 0.0, 0.0, 0.739, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.743, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.003, 0.0, 0.0, 0.0, 0.0, 0.751, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.757, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.761, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.769, 0.0, 0.0, 0.0,
             0.773, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_not_trainable_variable_variable.
                __func__,
            layer_factory.function[key])
        evaluated = layer_factory.build_atom(atom)(
            tf.eye(layer_factory.constant_size))

        self.assertEqual(
            (layer_factory.constant_size, layer_factory.constant_size),
            evaluated.shape)
        for i in range(layer_factory.constant_size):
            for j in range(layer_factory.constant_size):
                self.assertAlmostEqual(
                    correct[i, j], evaluated[i, j], places=EQUAL_DELTA,
                    msg="Incorrect value for constants {}, {}: {}, {}".format(
                        i, j, layer_factory.program.iterable_constants[i],
                        layer_factory.program.iterable_constants[j]))

    def test_arity_2_2_not_trainable_same_variables(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["X", "X"]
        atom = Atom("father", *value)
        correct = np.array([0.0, 0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 0.003, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_not_trainable_variable_variable.
                __func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            self.assertAlmostEqual(
                correct[i], evaluated[i], places=EQUAL_DELTA,
                msg="Incorrect value for constants {}: {}".format(
                    i, layer_factory.program.iterable_constants[i]))

    def test_arity_2_2_trainable_constant_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["some_male", "some_female"]
        atom = Atom("uncle", *value, weight=0.049)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_trainable_constant_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated, places=EQUAL_DELTA)

    def test_arity_2_2_trainable_constant_constant_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["some_male", "some_female2"]
        atom = Atom("uncle", *value, weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_trainable_constant_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertLessEqual(evaluated, atom.weight)
        self.assertGreaterEqual(evaluated, WEIGHT_MIN_VALUE)

    def test_arity_2_2_trainable_constant_not_in_kb_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["some_male2", "some_female"]
        atom = Atom("uncle", *value, weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_trainable_constant_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertLessEqual(evaluated, atom.weight)
        self.assertGreaterEqual(evaluated, WEIGHT_MIN_VALUE)

    def test_arity_2_2_trainable_constant_not_in_kb_constant_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["some_male2", "some_female2"]
        atom = Atom("uncle", *value, weight=WEIGHT_MAX_VALUE)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_trainable_constant_constant.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((), evaluated.shape)
        self.assertLessEqual(evaluated, atom.weight)
        self.assertGreaterEqual(evaluated, WEIGHT_MIN_VALUE)

    def test_arity_2_2_trainable_constant_iterable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["some_female", "charlotte"]
        atom = Atom("aunt", *value, weight=0.33)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_trainable_constant_iterable_constant.
                __func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((1, 1), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated, places=EQUAL_DELTA)

    def test_arity_2_2_trainable_constant_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["some_female", "X"]
        atom = Atom("aunt", *value)
        weight = np.array([0.21, 0.0, 0.0, 0.31, 0.33, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_trainable_constant_variable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            error_message = "Incorrect value for constant {}: {}".format(
                i, layer_factory.program.iterable_constants[i])
            if np.isclose(weight[i], 0.0):
                self.assertGreaterEqual(evaluated[i], WEIGHT_MIN_VALUE,
                                        msg=error_message)
                self.assertLessEqual(evaluated[i], WEIGHT_MAX_VALUE,
                                     msg=error_message)
            else:
                self.assertAlmostEqual(
                    weight[i], evaluated[i], places=EQUAL_DELTA,
                    msg=error_message)

    def test_arity_2_2_trainable_constant_not_in_kb_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["some_female2", "X"]
        atom = Atom("aunt", *value)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_trainable_constant_variable.__func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            error_message = "Incorrect value for constant {}: {}".format(
                i, layer_factory.program.iterable_constants[i])
            self.assertGreaterEqual(evaluated[i], WEIGHT_MIN_VALUE,
                                    msg=error_message)
            self.assertLessEqual(evaluated[i], WEIGHT_MAX_VALUE,
                                 msg=error_message)

    def test_arity_2_2_trainable_iterable_constant_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["jennifer", "some_male"]
        atom = Atom("aunt", *value, weight=0.0433)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_trainable_iterable_constant_constant.
                __func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((1, 1), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated, places=EQUAL_DELTA)

    def test_arity_2_2_trainable_iterable_constant_iterable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["jennifer", "charlotte"]
        atom = Atom("aunt", *value, weight=0.433)
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.
                arity_2_2_trainable_iterable_constant_iterable_constant.
                __func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((1, 1), evaluated.shape)
        self.assertAlmostEqual(atom.weight, evaluated[0], places=EQUAL_DELTA)

    def test_arity_2_2_trainable_iterable_constant_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["alfonso", "X"]
        atom = Atom("nephew", *value)
        weight = np.array([0.0, 0.0, 0.73, 0.079, 0.083, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_trainable_iterable_constant_variable.
                __func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((1, layer_factory.constant_size), evaluated.shape)
        for i in range(layer_factory.constant_size):
            error_message = "Incorrect value for constant {}: {}".format(
                i, layer_factory.program.iterable_constants[i])
            if np.isclose(weight[i], 0.0):
                self.assertGreaterEqual(evaluated[0][i], WEIGHT_MIN_VALUE,
                                        msg=error_message)
                self.assertLessEqual(evaluated[0][i], WEIGHT_MAX_VALUE,
                                     msg=error_message)
            else:
                self.assertAlmostEqual(
                    weight[i], evaluated[0][i], places=EQUAL_DELTA,
                    msg=error_message)

    def test_arity_2_2_trainable_variable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["X", "some_female"]
        atom = Atom("niece", *value)
        weight = np.array([0.0, 0.0, 0.0113, 0.0, 0.0127, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0107, 0.0109, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_trainable_variable_constant.__func__,
            layer_factory.function[key])
        input_value = tf.eye(layer_factory.constant_size)
        tensor = layer_factory.build_atom(atom)(input_value)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size, 1), evaluated.shape)
        for i in range(layer_factory.constant_size):
            error_message = "Incorrect value for constant {}: {}".format(
                i, layer_factory.program.iterable_constants[i])
            if np.isclose(weight[i], 0.0):
                self.assertGreaterEqual(evaluated[i], WEIGHT_MIN_VALUE,
                                        msg=error_message)
                self.assertLessEqual(evaluated[i], WEIGHT_MAX_VALUE,
                                     msg=error_message)
            else:
                self.assertAlmostEqual(weight[i], evaluated[i],
                                       places=EQUAL_DELTA, msg=error_message)

    def test_arity_2_2_trainable_variable_constant_not_in_kb(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["X", "some_female2"]
        atom = Atom("niece", *value)
        weight = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_trainable_variable_constant.__func__,
            layer_factory.function[key])
        input_value = tf.eye(layer_factory.constant_size)
        tensor = layer_factory.build_atom(atom)(input_value)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size, 1), evaluated.shape)
        for i in range(layer_factory.constant_size):
            error_message = "Incorrect value for constant {}: {}".format(
                i, layer_factory.program.iterable_constants[i])
            if np.isclose(weight[i], 0.0):
                self.assertGreaterEqual(evaluated[i], WEIGHT_MIN_VALUE,
                                        msg=error_message)
                self.assertLessEqual(evaluated[i], WEIGHT_MAX_VALUE,
                                     msg=error_message)
            else:
                self.assertAlmostEqual(
                    weight[i], evaluated[i], places=EQUAL_DELTA,
                    msg=error_message)

    def test_arity_2_2_trainable_variable_iterable_constant(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["X", "jennifer"]
        atom = Atom("niece", *value)
        weight = np.array([0.0, 0.0, 0.0131, 0.0, 0.113, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0149,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_trainable_variable_iterable_constant.
                __func__,
            layer_factory.function[key])
        input_value = tf.eye(layer_factory.constant_size)
        tensor = layer_factory.build_atom(atom)(input_value)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size, 1), evaluated.shape)
        for i in range(layer_factory.constant_size):
            error_message = "Incorrect value for constant {}: {}".format(
                i, layer_factory.program.iterable_constants[i])
            if np.isclose(weight[i], 0.0):
                self.assertGreaterEqual(evaluated[i], WEIGHT_MIN_VALUE,
                                        msg=error_message)
                self.assertLessEqual(evaluated[i], WEIGHT_MAX_VALUE,
                                     msg=error_message)
            else:
                self.assertAlmostEqual(
                    weight[i], evaluated[i], places=EQUAL_DELTA,
                    msg=error_message)

    def test_arity_2_2_trainable_variable_variable(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["X", "Y"]
        atom = Atom("nephew", *value)
        weight = np.array([
            [0.0, 0.0, 0.730, 0.079, 0.083, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.089, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.101,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_trainable_variable_variable.
                __func__,
            layer_factory.function[key])
        evaluated = layer_factory.build_atom(atom)(
            tf.eye(layer_factory.constant_size))

        self.assertEqual(
            (layer_factory.constant_size, layer_factory.constant_size),
            evaluated.shape)
        for i in range(layer_factory.constant_size):
            for j in range(layer_factory.constant_size):
                error_message = \
                    "Incorrect value for constants {}, {}: {}, {}".format(
                        i, j, layer_factory.program.iterable_constants[i],
                        layer_factory.program.iterable_constants[j])
                if np.isclose(weight[i, j], 0.0):
                    self.assertGreaterEqual(evaluated[i, j], WEIGHT_MIN_VALUE,
                                            msg=error_message)
                    self.assertLessEqual(evaluated[i, j], WEIGHT_MAX_VALUE,
                                         msg=error_message)
                else:
                    self.assertAlmostEqual(
                        weight[i, j], evaluated[i, j], places=EQUAL_DELTA,
                        msg=error_message)

    def test_arity_2_2_trainable_same_variables(self):
        layer_factory = self.layer_factory  # type: LayerFactory
        value = ["X", "X"]
        atom = Atom("uncle", *value)
        weight = np.array([0.0, 0.0, 0.0, 0.029, 0.0, 0.0, 0.0, 0.0, 0.013,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0])
        key = layer_factory.get_atom_key(atom)
        self.assertEqual(
            layer_factory.arity_2_2_trainable_variable_variable.
                __func__,
            layer_factory.function[key])
        tensor = layer_factory.build_atom(atom)(NEUTRAL_ELEMENT)

        evaluated = tensor.numpy()
        self.assertEqual((layer_factory.constant_size,), evaluated.shape)
        for i in range(layer_factory.constant_size):
            error_message = "Incorrect value for constants {}: {}".format(
                i, layer_factory.program.iterable_constants[i])
            if np.isclose(weight[i], 0.0):
                self.assertGreaterEqual(evaluated[i], WEIGHT_MIN_VALUE,
                                        msg=error_message)
                self.assertLessEqual(evaluated[i], WEIGHT_MAX_VALUE,
                                     msg=error_message)
            else:
                self.assertAlmostEqual(weight[i], evaluated[i],
                                       places=EQUAL_DELTA, msg=error_message)


if __name__ == '__main__':
    unittest.main()
