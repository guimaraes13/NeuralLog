"""
Handles the communication between the structure learning algorithm and the
inference engine.
"""
import collections
import logging
import os
import sys
from abc import abstractmethod
from typing import Optional, List, Set

import numpy as np

import src.structure_learning.structure_learning_system as sls
from src.knowledge.examples import Examples, ExamplesInferences, ExampleIterator
from src.knowledge.program import NeuralLogProgram, print_neural_log_program
from src.language.language import Atom, Term, Clause
from src.network.network import NeuralLogNetwork
from src.network.trainer import Trainer
from src.util import Initializable, time_measure
from src.util.file import read_logic_program_from_file

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_FILE_NAME = "knowledge_base.pl"
THEORY_FILE_NAME = "theory.pl"
SAVED_MODEL_FILE_NAME = "saved_model"

TEMPORARY_SET_NAME = "__temporary_set__"
ALL_TERMS_NAME = "__all_terms__"
NULL_SET_NAME = "__null_set__"


def append_theory(program, theory):
    """
    Appends the theory to the program.

    :param program: the program
    :type program: NeuralLogProgram
    :param theory: the theory
    :type theory: NeuralLogProgram or collections.Iterable[Clause]
    """
    if theory is not None:
        if isinstance(theory, NeuralLogProgram):
            for clauses in theory.clauses_by_predicate.values():
                program.add_clauses(clauses)
            program.add_clauses(theory.builtin_facts)
        else:
            program.add_clauses(theory)


# noinspection PyTypeChecker,DuplicatedCode
def convert_predictions(model, dataset, positive_threshold=None):
    """
    Gets the examples inferred by the `model`, based on the `dataset`.

    :param model: the model
    :type model: NeuralLogNetwork
    :param dataset: the dataset
    :type dataset: tf.data.Dataset
    :param positive_threshold: if set, only the examples whose inference are
    above the threshold will be considered as positive.
    :type positive_threshold: Optional[float]
    :return: the example inferences
    :rtype: ExamplesInferences
    """
    neural_program = model.program
    inferences = ExamplesInferences()
    empty_entry = None
    for features, labels in dataset:
        y_scores = model.predict(features)
        if y_scores is None:
            continue
        if len(model.predicates) == 1:
            y_scores = [y_scores]
            if model.predicates[0][0].arity < 3:
                features = [features]
        for i in range(len(model.predicates)):
            # Iterate over predicates
            predicate, inverted = model.predicates[i]
            null_index = neural_program.get_index_of_constant(
                predicate, predicate.arity - 1, sls.NULL_ENTITY)
            if inverted:
                continue
            row_scores = y_scores[i]
            if len(row_scores.shape) == 3:
                row_scores = np.squeeze(row_scores, axis=1)
            if predicate.arity > 1 and row_scores.shape[-1] == 1:
                # The network prediction is a scalar, this is, the prediction
                #   is equal for any output subject.
                #   As such, we can resize the output to represent this,
                #   as the line bellow:
                #   row_scores = np.resize(row_scores, labels[i].shape)
                # Since this function only return the examples with score
                #   greater than the null output, and since all the outputs
                #   have the same score, no output for this predicate will be
                #   greater than the null score, thus, it can break here as
                #   long as the positive_threshold is `None`.
                if positive_threshold is None:
                    break
                else:
                    row_scores = np.resize(row_scores, labels[i].shape)
            initial_offset = sum(model.input_sizes[:i])
            for j in range(len(row_scores)):
                # iterates over subjects
                if predicate.arity == 1:
                    if positive_threshold is None:
                        null_score = float(row_scores[null_index])
                    else:
                        null_score = positive_threshold
                    if float(row_scores[j]) <= null_score or j == null_index:
                        continue
                y_score = row_scores[j]
                x = []
                subjects = []
                stop = False
                offset = initial_offset
                for k in range(model.input_sizes[i]):
                    x_k = features[offset + k][j].numpy()
                    if x_k.dtype == np.float32:
                        if np.max(x_k) == 0:
                            stop = True
                            break
                        arg_max = np.argmax(x_k)
                        if arg_max == empty_entry:
                            stop = True
                            break
                    else:
                        arg_max = x_k[0]
                        if arg_max < 0 or arg_max == empty_entry:
                            stop = True
                            break
                    subjects.append(neural_program.get_constant_by_index(
                        predicate, k, arg_max))
                    x.append(x_k)
                offset += model.input_sizes[i]
                if stop:
                    continue

                if predicate.arity == 1:
                    atom = Atom(predicate, subjects[0], weight=float(y_score))
                    inferences.add_inference(atom)
                else:
                    # null_index = neural_program.get_index_of_constant(
                    #     predicate, -1, sls.NULL_ENTITY)
                    if positive_threshold is None:
                        null_score = float(y_score[null_index])
                    else:
                        null_score = positive_threshold
                    for index in range(len(y_score)):
                        atom_score = float(y_score[index])
                        if atom_score <= null_score or index == null_index:
                            continue
                        object_term = neural_program.get_constant_by_index(
                            predicate, -1, index)
                        atom = Atom(predicate, *subjects, object_term,
                                    weight=atom_score)
                        inferences.add_inference(atom)

    return inferences


class EngineSystemTranslator(Initializable):
    """
    Translates the results of the engine system to the structure learning
    algorithm and vice versa.
    """

    def __init__(self, knowledge_base=None, theory=None, output_path=None):
        """
        Creates the engine system translator.

        :param knowledge_base: the knowledge base
        :type knowledge_base: Optional[NeuralLogProgram]
        :param theory: the theory
        :type theory: Optional[NeuralLogProgram]
        :param output_path: the output path
        :type output_path: Optional[str]
        """
        self._knowledge_base: Optional[NeuralLogProgram] = None
        self._theory: Optional[NeuralLogProgram] = None
        self._output_path: Optional[str] = None
        self.knowledge_base = knowledge_base
        self.theory = theory
        self._output_path = output_path

    # noinspection PyMissingOrEmptyDocstring
    @property
    def knowledge_base(self):
        return getattr(self, "_knowledge_base", None)

    @knowledge_base.setter
    def knowledge_base(self, value):
        self._knowledge_base = value

    # noinspection PyMissingOrEmptyDocstring
    @property
    def theory(self):
        return getattr(self, "_theory", None)

    @theory.setter
    def theory(self, value):
        self._theory = value

    # noinspection PyMissingOrEmptyDocstring
    @property
    def output_path(self):
        return self._output_path

    @output_path.setter
    def output_path(self, value):
        self._output_path = value

    @abstractmethod
    def build_model(self):
        """
        Builds the model based on the current knowledge base and the theory.
        This method must be called whenever the knowledge base or the theory
        change.
        """
        pass

    @abstractmethod
    def infer_examples(self, examples, retrain=False, theory=None,
                       positive_threshold=None):
        """
        Perform the inference for the given examples. If `retrain` is `True`,
        it retrains the parameters before inference. If `theory` is not
        `None`, uses it as the theory to be evaluated, instead of the current
        theory of the engine system.

        :param examples: the examples
        :type examples: Examples or Atom
        :param retrain: if `True`, it retrains the parameters before inference
        :type retrain: bool
        :param theory: If not `None`, use this theory instead of the current
        theory of the engine system
        :type theory: NeuralLogProgram or collections.Iterable[Clause] or None
        :param positive_threshold: if set, only the examples whose inference
        are above the threshold will be considered as positive. If not set,
        only the examples whose score is above the score of the `__null__`
        example will be considered as positive
        :type positive_threshold: Optional[float]
        :return: the inference value of the examples
        :rtype: ExamplesInferences
        """
        pass

    @abstractmethod
    def infer_examples_appending_clauses(
            self, examples, clauses, retrain=False, positive_threshold=None):
        """
        Perform the inference for the given examples. The `clauses` are
        appended to the current theory, before evaluation. If `retrain` is
        `True`, it retrains the parameters before inference.

        After evaluation, the appended clauses are discarded and the parameters
        are restored to its values previously to the evaluation.

        :param examples: the examples
        :type examples: Examples
        :param retrain: if `True`, it retrains the parameters before inference
        :type retrain: bool
        :param clauses: clauses to be appended to the current theory, for
        evaluation proposes only
        :type clauses: collections.Iterable[Clause]
        :param positive_threshold: if set, only the examples whose inference
        are above the threshold will be considered as positive. If not set,
        only the examples whose score is above the score of the `__null__`
        example will be considered as positive
        :type positive_threshold: Optional[float]
        :return: the inference value of the examples
        :rtype: ExamplesInferences
        """
        pass

    @abstractmethod
    def inferred_relevant(self, terms, positive_threshold=None):
        """
        Perform the inference in order to get all atoms directly
        relevant to `terms`. The atoms directly relevant ot a term is the atoms
        which contain the term.

        :param terms: the terms
        :type terms: Set[Term]
        :param positive_threshold: if set, only the examples whose inference
        are above the threshold will be considered as positive. If not set,
        only the examples whose score is above the score of the `__null__`
        example will be considered as positive
        :type positive_threshold: Optional[float]
        :return: the atoms relevant to the terms
        :rtype: Set[Atom]
        """
        pass

    @abstractmethod
    def train_parameters(self, training_examples):
        """
        Trains the parameters of the model.

        :param training_examples: the training examples
        :type training_examples: Examples
        """
        pass

    @abstractmethod
    def save_trained_parameters(self):
        """
        Saves the trained parameters.
        """
        pass

    @abstractmethod
    def save_parameters(self, working_directory):
        """
        Saves the parameters to the working directory.

        :param working_directory: the working directory.
        :type working_directory: str
        """
        pass

    @abstractmethod
    def load_parameters(self, working_directory):
        """
        Loads the parameters from the working directory.

        :param working_directory: the working directory.
        :type working_directory: str
        """
        pass


class NeuralLogEngineSystemTranslator(EngineSystemTranslator):
    """
    A engine system translator for the NeuralLog language.
    """

    OPTIONAL_FIELDS = {
        "batch_size": 16,
        "_last_saved_change": -sys.float_info.max,
        "_last_relevant_inference": -sys.float_info.max,
        "_cached_inference": None
    }

    def __init__(self, knowledge_base=None, theory=None, output_path=None):
        super().__init__(knowledge_base, theory, output_path)
        self._saved_trainer: Optional[Trainer] = None
        self._last_saved_change: float = -sys.float_info.max
        self._last_relevant_inference: float = -sys.float_info.max
        self._cached_inference: Optional[NeuralLogProgram] = None
        self.current_trainer: Optional[Trainer] = None
        self.batch_size = self.OPTIONAL_FIELDS["batch_size"]

    @property
    def saved_trainer(self):
        """
        Gets the saved trainer.

        :return: the saved trainer
        :rtype: Trainer
        """
        return self._saved_trainer

    @saved_trainer.setter
    def saved_trainer(self, value):
        """
        Sets the saved trainer.

        :param value: the value
        :type value: Trainer
        """
        self._saved_trainer = value
        self._last_saved_change = time_measure.performance_time()

    # noinspection PyMissingOrEmptyDocstring
    @EngineSystemTranslator.knowledge_base.setter
    def knowledge_base(self, value: NeuralLogProgram):
        self._knowledge_base = value.copy()
        self.knowledge_base.parameters.setdefault("inverse_relations", False)
        self.build_model()

    # noinspection PyMissingOrEmptyDocstring
    @EngineSystemTranslator.theory.setter
    def theory(self, value):
        self._theory = value
        self.build_model()

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return []

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        self.build_model()

    def build_model(self):
        """
        Builds the model based on the current knowledge base and the theory.
        This method must be called whenever the knowledge base or the theory
        change.
        """
        if not self.knowledge_base or not self.theory:
            return
        program = self.knowledge_base.copy()
        append_theory(program, self.theory)
        self.saved_trainer = Trainer(program, self.output_path)
        self.saved_trainer.init_model()
        self.current_trainer = self.saved_trainer

    # noinspection PyMissingOrEmptyDocstring
    def infer_examples(self, examples, retrain=False, theory=None,
                       positive_threshold=None):
        trainer = self.get_trainer(examples, theory)

        dataset = trainer.build_dataset()
        dataset = dataset.get_dataset(TEMPORARY_SET_NAME, self.batch_size)
        if retrain:
            trainer.compile_module()
            trainer.fit(dataset)

        return convert_predictions(trainer.model, dataset, positive_threshold)

    def get_trainer(self, examples, theory=None):
        """
        Gets a trainer for the examples and the theory.

        :param examples: the examples
        :type examples: Examples or Atom
        :param theory: the theory
        :type theory: NeuralLogProgram or collections.Iterable[Clause] or None
        :return: the trainer
        :rtype: Trainer
        """
        program = self.knowledge_base.copy()
        if isinstance(examples, Examples):
            program.add_examples(examples, TEMPORARY_SET_NAME)
            for predicate in examples:
                null_atom = sls.build_null_atom(program, predicate)
                program.add_example(null_atom, NULL_SET_NAME, False)
        else:
            program.add_example(examples, TEMPORARY_SET_NAME)
            null_atom = sls.build_null_atom(program, examples.predicate)
            program.add_example(null_atom, NULL_SET_NAME, False)
        if theory is None:
            theory = self.theory
        append_theory(program, theory)
        program.build_program()
        trainer = Trainer(program, self.output_path)
        trainer.init_model()
        if isinstance(examples, Examples):
            trainer.model.build_layers(
                map(lambda x: (x, False), examples.keys()))
        else:
            trainer.model.build_layers([(examples.predicate, False)])
        trainer.read_parameters()
        return trainer

    # noinspection PyMissingOrEmptyDocstring
    def infer_examples_appending_clauses(
            self, examples, clauses, retrain=False, positive_threshold=None):
        # noinspection PyTypeChecker
        appended_clauses: List[Clause] = list()
        for theory_clauses in self.theory.clauses_by_predicate.values():
            appended_clauses.extend(theory_clauses)
        appended_clauses.extend(clauses)
        return self.infer_examples(
            examples, retrain=retrain,
            theory=appended_clauses, positive_threshold=positive_threshold)

    # noinspection PyMissingOrEmptyDocstring
    def inferred_relevant(self, terms, positive_threshold=None):
        if self._last_relevant_inference < self._last_saved_change or \
                self._cached_inference is None:
            self._update_example_terms()
            self.saved_trainer.read_parameters()
            dataset = \
                self.saved_trainer.neural_dataset.get_dataset(ALL_TERMS_NAME)
            self._cached_inference = NeuralLogProgram()
            if dataset is not None:
                inferences = convert_predictions(
                    self.saved_trainer.model, dataset, positive_threshold)
                self._cached_inference = NeuralLogProgram()
                for atom in ExampleIterator(inferences.examples):
                    self._cached_inference.add_fact(atom)
            self._last_relevant_inference = time_measure.performance_time()
        results = set()
        for term in terms:
            results.update(self._cached_inference.get_atoms_with_term(term))

        return results

    def _update_example_terms(self):
        layers = []
        program = self.saved_trainer.neural_program
        for predicate in program.clauses_by_predicate:
            possible_terms = []
            for i in range(predicate.arity - 1):
                terms = program.iterable_constants_per_term.get(
                    (predicate, i), dict()).values()
                possible_terms.append(terms)
            possible_terms.append((sls.NULL_ENTITY,))

            has_example = False
            for terms in permute_terms(*possible_terms):
                atom = Atom(predicate, *terms, weight=0.0)
                program.add_example(atom, ALL_TERMS_NAME, False)
                has_example = True
            if has_example:
                layers.append((predicate, False))

        self.saved_trainer.model.build_layers(layers)

    # noinspection PyMissingOrEmptyDocstring
    def train_parameters(self, training_examples):
        trainer = self.get_trainer(training_examples)
        trainer.compile_module()
        if trainer.model.has_trainable_parameters:
            dataset = trainer.build_dataset()
            dataset = dataset.get_dataset(TEMPORARY_SET_NAME)
            trainer.fit(dataset)
        self.current_trainer = trainer

    # noinspection PyMissingOrEmptyDocstring
    def save_trained_parameters(self):
        self.current_trainer.model.update_program(self.knowledge_base)
        self.saved_trainer = self.current_trainer

    # noinspection PyMissingOrEmptyDocstring
    def save_parameters(self, working_directory):
        logger.debug("Saving the trained model to path:\t%s", working_directory)
        filepath = os.path.join(working_directory, SAVED_MODEL_FILE_NAME)
        self.saved_trainer.model.save_weights(filepath)
        knowledge_file = \
            open(os.path.join(working_directory, KNOWLEDGE_BASE_FILE_NAME), "w")
        print_neural_log_program(self.knowledge_base, knowledge_file)
        knowledge_file.close()
        theory_file = \
            open(os.path.join(working_directory, THEORY_FILE_NAME), "w")
        print_neural_log_program(self.theory, theory_file)
        theory_file.close()

    # noinspection PyMissingOrEmptyDocstring
    def load_parameters(self, working_directory):
        logger.debug(
            "Loading the trained model from path:\t%s", working_directory)
        knowledge_base_path = \
            os.path.join(working_directory, KNOWLEDGE_BASE_FILE_NAME)
        theory_path = os.path.join(working_directory, THEORY_FILE_NAME)
        self.knowledge_base = NeuralLogProgram()
        clauses = read_logic_program_from_file(knowledge_base_path)
        self.knowledge_base.add_clauses(clauses)
        self.theory = NeuralLogProgram()
        clauses = read_logic_program_from_file(theory_path)
        self.theory.add_clauses(clauses)


def permute_terms(*iterables):
    """
    Yields the permutation of all items in `iterables`.

    :param iterables: the iterables
    :type iterables: List[collections.Iterable]
    :return: the permutation of all terms
    :rtype: generator
    """
    for item in iterables[0]:
        if len(iterables) > 1:
            remaining = permute_terms(*iterables[1:])
            for suffix in remaining:
                yield (item,) + suffix
        else:
            yield item,
