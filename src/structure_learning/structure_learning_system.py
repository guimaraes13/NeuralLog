"""
The core of the structure learning system.
"""

from src.knowledge.manager.example_manager import IncomingExampleManager
from src.knowledge.program import NeuralLogProgram
from src.knowledge.theory.evaluation.theory_evaluator import TheoryEvaluator
from src.knowledge.theory.manager.theory_revision_manager import \
    TheoryRevisionManager
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
        self.theory = theory
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
