"""
Tests the NeuralLog dataset.
"""
import logging
import os
import unittest
from typing import List

from neurallog.knowledge.program import NeuralLogProgram, NO_EXAMPLE_SET
from neurallog.language.language import HornClause, Predicate, Constant
from neurallog.language.parser.ply.neural_log_parser import NeuralLogLexer, \
    NeuralLogParser
from neurallog.network.dataset import LanguageDataset
from neurallog.network.trainer import Trainer
from neurallog.run import configure_log

RESOURCE_PATH = os.path.dirname(os.path.realpath(__file__))
VOCABULARY_FILE = os.path.join(RESOURCE_PATH, "vocab.txt")

POSSIBLE_LABELS = ["[PAD]", "B", "I", "O", "X", "[CLS]", "[SEP]"]

# noinspection SpellCheckingInspection
TOKENS = [
    "Clustering",
    "of",
    "missense",
    "mutations",
    "in",
    "the",
    "ataxia",
    "-",
    "telangiectasia",
    "gene",
    "in",
    "a",
    "sporadic",
    "T",
    "-",
    "cell",
    "leukaemia",
    ".",
]

EXPANDED_TOKENS = [
    ("[CLS]", "[CLS]"),
    ("C", "O"),
    ("##luster", "X"),
    ("##ing", "X"),
    ("of", "O"),
    ("miss", "O"),
    ("##ense", "X"),
    ("mutations", "O"),
    ("in", "O"),
    ("the", "O"),
    ("at", "O"),
    ("##ax", "X"),
    ("##ia", "X"),
    ("-", "O"),
    ("te", "O"),
    ("##lang", "X"),
    ("##ie", "X"),
    ("##ct", "X"),  # Included in 18
    ("##asi", "X"),
    ("##a", "X"),  # Included in 20
    ("gene", "O"),
    ("in", "O"),  # Included in 22
    ("a", "O"),
    ("s", "O"),
    ("##poradic", "X"),
    ("T", "O"),
    ("-", "O"),
    ("cell", "O"),
    ("le", "O"),
    ("##uka", "X"),
    ("##emia", "X"),
    (".", "O"),
    ("[SEP]", "[SEP]"),
]

EXAMPLE = \
    "\n".join(map(lambda x: f"mega_example(0, ner, \"{x}\", \"O\").", TOKENS))

possible_labels = "\n".join(map(lambda x: f'label("{x}").', POSSIBLE_LABELS))
# noinspection SpellCheckingInspection
PROGRAM = f"""
word("<OOV>").
word("Clustering").
word("of").
word("missense").
word("mutations").
word("in").
word("the").
word("ataxia").
word("-").
word("telangiectasia").
word("gene").
word("a").
word("sporadic").
word("T").
word("cell").
word("leukaemia").
word(".").

{possible_labels}

type_ner(X, Y):- ner(X, Y), word(X), label(Y).

ner(X, Y) :- true.

set_parameter(inverse_relations, false).
""".strip()

SEQUENCE_DATASET = """
set_parameter(dataset_class, class_name, sequence_dataset).
set_parameter(dataset_class, config, oov_word, "<OOV>").
set_parameter(dataset_class, config, expand_one_hot, "False").
set_parameter(dataset_class, config, empty_word_index, -1).
""".strip()

LANGUAGE_DATASET = f"""
set_parameter(dataset_class, class_name, language_dataset).
set_parameter(dataset_class, config, inverse_relations, false).
set_parameter(dataset_class, config, vocabulary_file, "{VOCABULARY_FILE}").
set_parameter(dataset_class, config, initial_token, "[CLS]").
set_parameter(dataset_class, config, final_token, "[SEP]").
set_parameter(dataset_class, config, pad_token, "[PAD]").
set_parameter(dataset_class, config, sub_token_label, "X").
set_parameter(dataset_class, config, maximum_sentence_length, {{maximum_len}}).
set_parameter(dataset_class, config, pad_to_maximum_length, {{pad_to_maximum}}).
set_parameter(dataset_class, config, do_lower_case, false).
""".strip()


def _read_program(program):
    """
    Reads the meta program.

    :return: the list of meta clauses
    :rtype: List[HornClause]
    """
    lexer = NeuralLogLexer()
    parser = NeuralLogParser(lexer)
    parser.parser.parse(input=program, lexer=lexer)
    parser.expand_placeholders()
    # noinspection PyTypeChecker
    return list(parser.get_clauses())


# noinspection DuplicatedCode
class TestSequenceDataset(unittest.TestCase):

    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        configure_log(level=logging.DEBUG)

    def test_sequence_dataset(self):
        program = NeuralLogProgram()
        program.add_clauses(_read_program(PROGRAM))
        program.add_clauses(_read_program(SEQUENCE_DATASET))
        program.add_clauses(_read_program(EXAMPLE))
        program.build_program()

        trainer = Trainer(program, output_path=None)
        trainer.init_model()
        trainer.read_parameters()
        neural_dataset = trainer.build_dataset(override_targets=False)
        result = list(neural_dataset.build(NO_EXAMPLE_SET))
        # dataset = neural_dataset.get_dataset(NO_EXAMPLE_SET)
        self.assertEqual(len(TOKENS), len(result[0][0][0]))
        self.assertEqual((len(TOKENS), len(POSSIBLE_LABELS)),
                         result[0][1][0].numpy().shape)

        target = Predicate("ner", 2)
        label_index = program.get_index_of_constant(target, 1, Constant("O"))
        for i, token_id in enumerate(result[0][0][0]):
            self.assertEqual(
                program.get_index_of_constant(target, 0, Constant(TOKENS[i])),
                token_id)
            self.assertEqual(label_index, result[0][1][0][i].numpy().argmax())

    def test_language_dataset(self):
        program = NeuralLogProgram()
        program.add_clauses(_read_program(PROGRAM))
        program.add_clauses(_read_program(
            LANGUAGE_DATASET.format(maximum_len=128, pad_to_maximum="false")))
        program.add_clauses(_read_program(EXAMPLE))
        program.build_program()

        trainer = Trainer(program, output_path=None)
        trainer.init_model()
        trainer.read_parameters()
        # noinspection PyTypeChecker
        neural_dataset: LanguageDataset = \
            trainer.build_dataset(override_targets=False)
        result = list(neural_dataset.build(NO_EXAMPLE_SET))
        # dataset = neural_dataset.get_dataset(NO_EXAMPLE_SET)
        self.assertEqual(len(EXPANDED_TOKENS), len(result[0][0][0]))
        self.assertEqual((len(EXPANDED_TOKENS), len(POSSIBLE_LABELS)),
                         result[0][1][0].numpy().shape)

        target = Predicate("ner", 2)
        expected_ids = neural_dataset.tokenizer.convert_tokens_to_ids(
            map(lambda x: x[0], EXPANDED_TOKENS))
        for i, token_id in enumerate(result[0][0][0]):
            self.assertEqual(expected_ids[i], token_id)
            label_index = program.get_index_of_constant(
                target, -1, Constant(EXPANDED_TOKENS[i][1]))
            self.assertEqual(label_index, result[0][1][0][i].numpy().argmax())

    def test_language_dataset_maximum_length(self):
        self.language_dataset_maximum_length(22)

    def test_language_dataset_maximum_length_end_sub_token(self):
        self.language_dataset_maximum_length(20)

    def test_language_dataset_maximum_length_middle_sub_token(self):
        self.language_dataset_maximum_length(18)

    def language_dataset_maximum_length(self, maximum_length):
        """
        Tests the language dataset with a maximum length.

        :param maximum_length: the maximum length
        :type maximum_length: int
        """
        program = NeuralLogProgram()
        program.add_clauses(_read_program(PROGRAM))
        program.add_clauses(_read_program(
            LANGUAGE_DATASET.format(maximum_len=maximum_length,
                                    pad_to_maximum="false")))
        program.add_clauses(_read_program(EXAMPLE), )
        program.build_program()
        trainer = Trainer(program, output_path=None)
        trainer.init_model()
        trainer.read_parameters()
        # noinspection PyTypeChecker
        neural_dataset: LanguageDataset = \
            trainer.build_dataset(override_targets=False)
        result = list(neural_dataset.build(NO_EXAMPLE_SET))
        # dataset = neural_dataset.get_dataset(NO_EXAMPLE_SET)
        self.assertEqual(maximum_length, len(result[0][0][0]))
        self.assertEqual(
            (maximum_length, len(POSSIBLE_LABELS)),
            result[0][1][0].numpy().shape)
        target = Predicate("ner", 2)
        expected_ids = neural_dataset.tokenizer.convert_tokens_to_ids(
            map(lambda x: x[0], EXPANDED_TOKENS))
        for i, token_id in enumerate(result[0][0][0][:maximum_length - 1]):
            self.assertEqual(expected_ids[i], token_id)
            label_index = program.get_index_of_constant(
                target, -1, Constant(EXPANDED_TOKENS[i][1]))
            self.assertEqual(label_index, result[0][1][0][i].numpy().argmax())
        self.assertEqual(expected_ids[-1], result[0][0][0][-1])
        label_index = program.get_index_of_constant(
            target, -1, Constant(EXPANDED_TOKENS[-1][1]))
        self.assertEqual(label_index, result[0][1][0][-1].numpy().argmax())

    def test_language_dataset_padding(self):
        program = NeuralLogProgram()
        program.add_clauses(_read_program(PROGRAM))
        maximum_length = 40
        program.add_clauses(_read_program(
            LANGUAGE_DATASET.format(maximum_len=maximum_length,
                                    pad_to_maximum="true")))
        program.add_clauses(_read_program(EXAMPLE), )
        program.build_program()
        trainer = Trainer(program, output_path=None)
        trainer.init_model()
        trainer.read_parameters()
        # noinspection PyTypeChecker
        neural_dataset: LanguageDataset = \
            trainer.build_dataset(override_targets=False)
        result = list(neural_dataset.build(NO_EXAMPLE_SET))
        # dataset = neural_dataset.get_dataset(NO_EXAMPLE_SET)
        self.assertEqual(maximum_length, len(result[0][0][0]))
        self.assertEqual((maximum_length, len(POSSIBLE_LABELS)),
                         result[0][1][0].numpy().shape)
        target = Predicate("ner", 2)
        expected_ids = neural_dataset.tokenizer.convert_tokens_to_ids(
            map(lambda x: x[0], EXPANDED_TOKENS))
        for i, token_id in enumerate(result[0][0][0][:len(EXPANDED_TOKENS)]):
            self.assertEqual(expected_ids[i], token_id)
            label_index = program.get_index_of_constant(
                target, 1, Constant(EXPANDED_TOKENS[i][1]))
            self.assertEqual(label_index, result[0][1][0][i].numpy().argmax())

        pad_feature = neural_dataset.tokenizer.vocab["[PAD]"]
        pad_label = program.get_index_of_constant(target, -1, Constant("[PAD]"))
        for i, token_id in enumerate(result[0][0][0][len(EXPANDED_TOKENS):]):
            i += len(EXPANDED_TOKENS)
            self.assertEqual(pad_feature, token_id)
            self.assertEqual(pad_label, result[0][1][0][i].numpy().argmax())
