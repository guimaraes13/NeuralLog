"""
The core of the structure learning system.
"""
from typing import Dict

from src.knowledge.manager.example_manager import IncomingExampleManager
from src.knowledge.program import NeuralLogProgram
from src.knowledge.theory.evaluation.theory_evaluator import TheoryEvaluator
from src.knowledge.theory.manager.theory_revision_manager import \
    TheoryRevisionManager
from src.language.language import Predicate, Atom
from src.structure_learning.engine_system_translator import \
    EngineSystemTranslator


class StructureLearningSystem:
    """
    Represents the core of the structure learning system.
    """

    def __init__(self, knowledge_base, theory, examples,
                 engine_system_translator,
                 theory_revision_manager, theory_evaluator,
                 incoming_example_manager):
        """
        Creates the structure learning system.

        :param knowledge_base: the knowledge base
        :type knowledge_base: NeuralLogProgram
        :param theory: the theory
        :type theory: NeuralLogProgram
        :param examples: the examples
        :type examples: NeuralLogProgram
        :param engine_system_translator: the engine system translator
        :type engine_system_translator: EngineSystemTranslator
        :param theory_revision_manager: the theory revision manager
        :type theory_revision_manager: TheoryRevisionManager
        :param theory_evaluator: the theory evaluator
        :type theory_evaluator: TheoryEvaluator
        :param incoming_example_manager: the incoming example manager
        :type incoming_example_manager: IncomingExampleManager
        """
        self.knowledge_base = knowledge_base
        "The knowledge base"
        self._theory = theory
        "The theory"
        self.examples = examples
        "The examples"

        self.engine_system_translator = engine_system_translator
        "The engine system translator"

        self.theory_revision_manager = theory_revision_manager
        "The theory revision manager"
        self.theory_evaluator = theory_evaluator
        "The theory evaluator"

        self.incoming_example_manager = incoming_example_manager
        "The incoming example manager"

    # noinspection PyMissingOrEmptyDocstring
    @property
    def theory(self):
        return self._theory

    @theory.setter
    def theory(self, value):
        self._theory = value
        # TODO: update the theory of the engine system translator

    def revise_theory(self, revision_examples):
        """
        Revises the theory based on the revision examples.

        :param revision_examples: the revision examples
        :type revision_examples:
            RevisionExamples or collections.Iterable[RevisionExamples]
        """
        self.theory_revision_manager.revise(revision_examples)

    def infer_examples(self, examples):
        """
        Perform the inference for the given examples.

        :param examples: the examples
        :type examples: Dict[Predicate, Dict[Any, Atom]]
        :return: the inference value of the examples
        :rtype: Dict[Predicate, Dict[Any, float]]
        """
        self.engine_system_translator.infer_examples(examples)

    def train_parameters(self, training_examples):
        """
        Trains the parameters of the model.

        :param training_examples: the training examples
        :type training_examples: Dict[Predicate, Dict[Any, Atom]]
        """
        self.engine_system_translator.train_parameters(training_examples)

    def save_trained_parameters(self):
        """
        Saves the trained parameters.
        """
        self.engine_system_translator.save_trained_parameters()
