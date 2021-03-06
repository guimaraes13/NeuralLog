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
The core of the structure learning system.
"""
import collections
from typing import Set, Dict

import neurallog.knowledge.manager.example_manager as manager
from neurallog.knowledge.examples import Examples, ExamplesInferences
from neurallog.knowledge.program import NeuralLogProgram
from neurallog.knowledge.theory.evaluation.metric.theory_metric import \
    TheoryMetric
from neurallog.knowledge.theory.evaluation.theory_evaluator import \
    TheoryEvaluator
from neurallog.knowledge.theory.manager.theory_revision_manager import \
    TheoryRevisionManager
from neurallog.language.language import get_term_from_string, Atom, Predicate, \
    Term
from neurallog.structure_learning.engine_system_translator import \
    EngineSystemTranslator
from neurallog.util import Initializable

NULL_ENTITY = get_term_from_string("'__NULL__'")


def build_null_atom(knowledge_base, predicate):
    """
    Builds the null atom for the predicate. The null atom is an atom with a
    constant, that must not appear anywhere else, for each logic term;
    zeros for the numeric terms; and zero weight.

    :param knowledge_base: the knowledge base
    :type knowledge_base: NeuralLogProgram
    :param predicate: the predicate
    :type predicate: Predicate
    :return: the null atom
    :rtype: Atom
    """
    terms = []
    for term_type in knowledge_base.predicates[predicate]:
        if term_type.number:
            terms.append(0.0)
        else:
            terms.append(NULL_ENTITY)

    return Atom(predicate, *terms, weight=0.0)


def add_null_atoms(knowledge_base):
    """
    Adds the null atoms to the knowledge base.

    :param knowledge_base: the knowledge base
    :type knowledge_base: NeuralLogProgram
    """
    for predicate in knowledge_base.predicates:
        null_atom = build_null_atom(knowledge_base, predicate)
        knowledge_base.add_fact(null_atom, report_replacement=False)
    # knowledge_base.build_program()


class StructureLearningSystem(Initializable):
    """
    Represents the core of the structure learning system.
    """

    def __init__(self, knowledge_base, theory, engine_system_translator,
                 theory_revision_manager=None, theory_evaluator=None,
                 incoming_example_manager=None):
        """
        Creates the structure learning system.

        :param knowledge_base: the knowledge base
        :type knowledge_base: NeuralLogProgram
        :param theory: the theory
        :type theory: NeuralLogProgram
        :param engine_system_translator: the engine system translator
        :type engine_system_translator: EngineSystemTranslator
        :param theory_revision_manager: the theory revision manager
        :type theory_revision_manager: TheoryRevisionManager or None
        :param theory_evaluator: the theory evaluator
        :type theory_evaluator: TheoryEvaluator or None
        :param incoming_example_manager: the incoming example manager
        :type incoming_example_manager: manager.IncomingExampleManager or None
        """
        self.engine_system_translator = engine_system_translator
        "The engine system translator"

        self.knowledge_base = knowledge_base
        "The knowledge base"

        self.theory = theory
        "The theory"

        self.theory_revision_manager = theory_revision_manager
        "The theory revision manager"

        self.theory_evaluator = theory_evaluator
        "The theory evaluator"

        self.incoming_example_manager = incoming_example_manager
        "The incoming example manager"

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return [
            "knowledge_base", "engine_system_translator", "theory_evaluator",
            "theory_revision_manager", "incoming_example_manager"
        ]

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()

        self.theory_revision_manager.learning_system = self
        self.theory_revision_manager.initialize()

        self.theory_evaluator.learning_system = self
        self.theory_evaluator.initialize()

        self.incoming_example_manager.learning_system = self
        self.incoming_example_manager.initialize()

    # noinspection PyMissingOrEmptyDocstring
    @property
    def knowledge_base(self) -> NeuralLogProgram:
        return self.engine_system_translator.knowledge_base

    @knowledge_base.setter
    def knowledge_base(self, value: NeuralLogProgram):
        self.engine_system_translator.knowledge_base = value

    @property
    def theory(self):
        """
        Gets the theory of the learning system.

        :return: the theory
        :rtype: NeuralLogProgram
        """
        return self.engine_system_translator.theory

    @theory.setter
    def theory(self, value):
        self.engine_system_translator.theory = value

    def revise_theory(self, revision_examples):
        """
        Revises the theory based on the revision examples.

        :param revision_examples: the revision examples
        :type revision_examples:
            RevisionExamples or collections.Collection[RevisionExamples]
        """
        self.theory_revision_manager.revise(revision_examples)

    def infer_examples(self, examples, theory=None,
                       retrain=False, positive_threshold=None):
        """
        Perform the inference for the given examples. If `theory` is not
        `None`, it is used theory as the theory to be evaluated, instead of the
        current theory of the engine system. If `retrain` is `True`,
        it retrains the parameters before inference.

        :param examples: the examples
        :type examples: Examples or Atom
        :param retrain: if `True`, it retrains the parameters before inference
        :type retrain: bool
        :param theory: If not `None`, use this theory instead of the current
        theory of the engine system.
        :type theory: Optional[NeuralLogProgram or collections.Iterable[Clause]]
        :param positive_threshold: if set, only the examples whose inference
        are above the threshold will be considered as positive. If not set,
        only the examples whose score is above the score of the `__null__`
        example will be considered as positive
        :type positive_threshold: Optional[float]
        :return: the inference value of the examples
        :rtype: ExamplesInferences
        """
        return self.engine_system_translator.infer_examples(
            examples, retrain, theory, positive_threshold=positive_threshold)

    def infer_examples_appending_clauses(
            self, examples, clauses, retrain=False, positive_threshold=None):
        """
        Perform the inference for the given examples. The `clauses` are
        appended to the current theory, before evaluation. If `retrain` is
        `True`, it retrains the parameters before inference.

        After evaluation, the appended clauses are discarded and the
        parameters are restored to its values previously to the evaluation.

        :param examples: the examples
        :type examples: Examples
        :param retrain: if `True`, it retrains the parameters before
        inference
        :type retrain: bool
        :param clauses: clauses to be appended to the current theory,
        for evaluation proposes only
        :type clauses: collections.Iterable[Clause]
        :param positive_threshold: if set, only the examples whose inference
        are above the threshold will be considered as positive. If not set,
        only the examples whose score is above the score of the `__null__`
        example will be considered as positive
        :type positive_threshold: Optional[float]
        :return: the inference value of the examples
        :rtype: ExamplesInferences
        """
        return self.engine_system_translator.infer_examples_appending_clauses(
            examples, clauses, retrain=retrain,
            positive_threshold=positive_threshold)

    def inferred_relevant(self, terms, positive_threshold=None):
        """
        Perform the inference in order to get all atoms directly relevant to
        `terms`. The atoms directly relevant ot a term is the atoms which
        contain the term.

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
        return self.engine_system_translator.inferred_relevant(
            terms, positive_threshold)

    def train_parameters(self, training_examples: Examples):
        """
        Trains the parameters of the model.

        :param training_examples: the training examples
        :type training_examples: Examples
        """
        self.engine_system_translator.train_parameters(training_examples)

    def save_trained_parameters(self):
        """
        Saves the trained parameters.
        """
        self.engine_system_translator.save_trained_parameters()

    def evaluate(self, examples, inferences):
        """
        Evaluates the examples based on the inferences.

        :param examples: the examples
        :type examples: Examples
        :param inferences: the inferences
        :type inferences: ExamplesInferences
        :return: a dictionary of evaluations per metric
        :rtype: Dict[TheoryMetric, float]
        """
        return self.theory_evaluator.evaluate(examples, inferences)

    def save_parameters(self, working_directory):
        """
        Saves the parameters to the working directory.

        :param working_directory: the working directory.
        :type working_directory: str
        """
        self.engine_system_translator.save_parameters(working_directory)

    def load_parameters(self, working_directory):
        """
        Loads the parameters from the working directory.

        :param working_directory: the working directory.
        :type working_directory: str
        """
        self.engine_system_translator.load_parameters(working_directory)
        self.knowledge_base = self.engine_system_translator.knowledge_base
        self.theory = self.engine_system_translator.theory
