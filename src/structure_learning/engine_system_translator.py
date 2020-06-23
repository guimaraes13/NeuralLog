"""
Handles the communication between the structure learning algorithm and the
inference engine.
"""
import collections
from abc import abstractmethod
from typing import Optional, List, Set

import numpy as np

from src.knowledge.examples import Examples, ExamplesInferences
from src.knowledge.program import NeuralLogProgram
from src.language.language import Atom, Term
from src.network.network import NeuralLogNetwork
from src.network.trainer import Trainer
from src.structure_learning.structure_learning_system import NULL_ENTITY
from src.util import Initializable

TEMPORARY_SET_NAME = "__temporary_set__"
ALL_TERMS_NAME = "__all_terms__"


class EngineSystemTranslator(Initializable):
    """
    Translates the results of the engine system to the structure learning
    algorithm and vice versa.
    """

    def __init__(self):
        self.knowledge_base: Optional[NeuralLogProgram] = None
        self.theory: Optional[NeuralLogProgram] = None
        self.output_path: Optional[str] = None

    @abstractmethod
    def infer_examples(self, examples, retrain=False, theory=None):
        """
        Perform the inference for the given examples. If `retrain` is `True`,
        it retrains the parameters before inference. If `theory` is not
        `None`, uses it as the theory to be evaluated, instead of the current
        theory of the engine system.

        :param examples: the examples
        :type examples: Examples
        :param retrain: if `True`, it retrains the parameters before inference
        :type retrain: bool
        :param theory: If not `None`, use this theory instead of the current
        theory of the engine system.
        :type theory: NeuralLogProgram
        :return: the inference value of the examples
        :rtype: ExamplesInferences
        """
        pass

    @abstractmethod
    def infer_examples_appending_clauses(self, examples, clauses,
                                         retrain=False):
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
        :type clauses: collections.Iterable[HornClauses]
        :return: the inference value of the examples
        :rtype: ExamplesInferences
        """
        pass

    @abstractmethod
    def inferred_relevant(self, terms):
        """
        Perform the inference in order to get all atoms directly
        relevant to `terms`. The atoms directly relevant ot a term is the atoms
        which contain the term.

        :param terms: the terms
        :type terms: Set[Term]
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


def append_theory(program, theory):
    """
    Appends the theory to the program.

    :param program: the program
    :type program: NeuralLogProgram
    :param theory: the theory
    :type theory: NeuralLogProgram or List[Clause]
    """
    if theory is not None:
        if isinstance(theory, NeuralLogProgram):
            for clauses in theory.clauses_by_predicate.values():
                program.add_clauses(clauses)
        else:
            program.add_clauses(theory)


# noinspection PyTypeChecker,DuplicatedCode
def convert_predictions(model, dataset):
    """
    Gets the examples inferred by the `model`, based on the `dataset`.

    :param model: the model
    :type model: NeuralLogNetwork
    :param dataset: the dataset
    :type dataset: tf.data.Dataset
    :return: the example inferences
    :rtype: ExamplesInferences
    """
    neural_program = model.program
    inferences = ExamplesInferences()
    empty_entry = None
    for features, _ in dataset:
        y_scores = model.predict(features)
        if len(model.predicates) == 1:
            y_scores = [y_scores]
            if model.predicates[0][0].arity < 3:
                features = [features]
        for i in range(len(model.predicates)):
            # Iterate over predicates
            predicate, inverted = model.predicates[i]
            null_index = neural_program.get_index_of_constant(
                predicate, -1, NULL_ENTITY)
            if inverted:
                continue
            row_scores = y_scores[i]
            if len(row_scores.shape) == 3:
                row_scores = np.squeeze(row_scores, axis=1)
            initial_offset = sum(model.input_sizes[:i])
            null_score = float(row_scores[null_index])
            for j in range(len(row_scores)):
                # iterates over subjects
                if predicate.arity == 1 and null_score >= float(row_scores[j]):
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
                    null_index = neural_program.get_constant_by_index(
                        predicate, -1, NULL_ENTITY)
                    null_score = float(y_score[null_index])
                    for index in range(len(y_score)):
                        atom_score = float(y_score[index])
                        if null_score >= atom_score:
                            continue
                        object_term = neural_program.get_constant_by_index(
                            predicate, -1, index)
                        atom = Atom(predicate, *subjects, object_term,
                                    weight=atom_score)
                        inferences.add_inference(atom)

    return inferences


class NeuralLogEngineSystemTranslator(EngineSystemTranslator):
    """
    A engine system translator for the NeuralLog language.
    """

    def __init__(self):
        super().__init__()
        self.saved_trainer: Optional[Trainer] = None
        self.current_trainer: Optional[Trainer] = None

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return []

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        program = self.knowledge_base.copy()
        for clauses in self.theory.clauses_by_predicate.values():
            program.add_clauses(clauses)
        program.build_program()
        self._build_model()

    def _build_model(self):
        program = self.knowledge_base.copy()
        append_theory(program, self.theory)
        self.saved_trainer = Trainer(program, self.output_path)
        self.saved_trainer.init_model()
        self.current_trainer = self.saved_trainer

    # IMPROVE: avoid rebuilding the program and the network whenever possible
    # noinspection PyMissingOrEmptyDocstring
    def infer_examples(self, examples, retrain=False, theory=None):
        trainer = self.get_trainer(examples, theory)

        dataset = trainer.build_dataset()
        dataset = dataset.get_dataset(TEMPORARY_SET_NAME)
        if retrain:
            trainer.fit(dataset)

        return convert_predictions(trainer.model, dataset)

    def get_trainer(self, examples, theory=None):
        """
        Gets a trainer for the examples and the theory.

        :param examples: the examples
        :type examples: Examples
        :param theory: the theory
        :type theory: Optional[NeuralLogProgram]
        :return:
        :rtype:
        """
        program = self.knowledge_base.copy()
        program.add_examples(examples, TEMPORARY_SET_NAME)
        if theory is None:
            theory = self.theory
        append_theory(program, theory)
        program.build_program()
        trainer = Trainer(program, None)
        trainer.init_model()
        trainer.model.build_layers(
            map(lambda x: (x, False), examples.keys()))
        trainer.compile_module()
        return trainer

    # noinspection PyMissingOrEmptyDocstring
    def infer_examples_appending_clauses(
            self, examples, clauses, retrain=False):
        # noinspection PyTypeChecker
        return self.infer_examples(examples, retrain=retrain, theory=clauses)

    # noinspection PyMissingOrEmptyDocstring
    def inferred_relevant(self, terms):
        self._update_example_terms()
        dataset = self.saved_trainer.neural_dataset.get_dataset(ALL_TERMS_NAME)
        inferences = convert_predictions(self.saved_trainer.model, dataset)
        examples = self.knowledge_base.examples.get(ALL_TERMS_NAME, Examples())
        results = set()
        for predicate, facts in examples.items():
            inferred = inferences.get(predicate)
            if inferred is None:
                continue
            for key, atom in facts.items():
                if key in inferred and not terms.isdisjoint(atom.terms):
                    results.add(atom)

        return results

    def _update_example_terms(self):
        layers = []
        for predicate in self.knowledge_base.clauses_by_predicate:
            layers.append((predicate, False))
            possible_terms = []
            for i in range(predicate.arity - 1):
                terms = self.knowledge_base.iterable_constants_per_term.get(
                    (predicate, i), dict()).values()
                possible_terms.append(terms)
            possible_terms.append((NULL_ENTITY,))

            for terms in permute_terms(*possible_terms):
                atom = Atom(predicate, *terms, weight=0.0)
                self.knowledge_base.add_example(atom, ALL_TERMS_NAME, False)
        self.saved_trainer.model.build_layers(layers)

    # noinspection PyMissingOrEmptyDocstring
    def train_parameters(self, training_examples):
        # IMPROVE: check if the program is up to date
        # IMPROVE: clean the examples after training
        trainer = self.get_trainer(training_examples)
        dataset = trainer.build_dataset()
        dataset = dataset.get_dataset(TEMPORARY_SET_NAME)
        trainer.fit(dataset)
        self.current_trainer = trainer

    # noinspection PyMissingOrEmptyDocstring
    def save_trained_parameters(self):
        self.current_trainer.model.update_program(self.knowledge_base)
        self.saved_trainer = self.current_trainer


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
