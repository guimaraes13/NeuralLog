"""
Handle the revision operators.
"""
import logging
from abc import abstractmethod
from collections import Collection
from typing import Dict

from src.knowledge.examples import Examples, ExampleIterator, ExamplesInferences
from src.knowledge.program import NeuralLogProgram
from src.knowledge.theory import TheoryRevisionException
from src.knowledge.theory.evaluation.metric.theory_metric import TheoryMetric
from src.knowledge.theory.evaluation.theory_evaluator import TheoryEvaluator
from src.knowledge.theory.manager.revision.clause_modifier import ClauseModifier
from src.language.language import KnowledgeException, Atom, Predicate, \
    HornClause
from src.structure_learning.structure_learning_system import \
    StructureLearningSystem, build_null_atom
from src.util import Initializable, InitializationException
from src.util.multiprocessing.evaluation_transformer import \
    EquivalentHonClauseAsyncTransformer
from src.util.multiprocessing.multiprocessing import MultiprocessingEvaluation
from src.util.variable_generator import VariableGenerator

DEFAULT_VARIABLE_GENERATOR = VariableGenerator

logger = logging.getLogger(__name__)

cached_null_atoms: Dict[Predicate, Atom] = dict()


def get_null_example(knowledge_base, predicate):
    """
    Gets and cached the null example for the predicate.

    :param knowledge_base: the knowledge base
    :type knowledge_base: NeuralLogProgram
    :param predicate: the predicate
    :type predicate: Predicate
    :return: the null example
    :rtype: Atom
    """
    null_atom = cached_null_atoms.get(predicate)
    if null_atom is None:
        null_atom = build_null_atom(knowledge_base, predicate)
        cached_null_atoms[predicate] = null_atom

    return null_atom


def is_covered(knowledge_base, example, inferred_examples):
    """
    Checks if the example is covered.

    :param knowledge_base: the knowledge base
    :type knowledge_base: NeuralLogProgram
    :param example: the example
    :type example: Atom
    :param inferred_examples: the inference of the examples
    :type inferred_examples: ExamplesInferences
    :return: `True`, if the example is covered; otherwise, `False`
    :rtype: bool
    """
    null_atom = get_null_example(knowledge_base, example.predicate)
    null_value = inferred_examples.get_value_for_example(null_atom)
    example_value = inferred_examples.get_value_for_example(example)

    return example_value > null_value


# TODO: extend this class
class RevisionOperator(Initializable):
    """
    Operator to revise the theory.
    """

    def __init__(self, learning_system=None, theory_metric=None,
                 clause_modifiers=None):
        """
        Creates a revision operator.

        :param learning_system: the learning system
        :type learning_system: StructureLearningSystem
        :param theory_metric: the theory metric
        :type theory_metric: TheoryMetric
        :param clause_modifiers: a clause modifier, a list of clause modifiers
        or none
        :type clause_modifiers: ClauseModifier or Collection[ClauseModifier]
        or None
        """
        self.learning_system = learning_system
        self.theory_metric = theory_metric
        self.clause_modifiers: Collection[ClauseModifier] = clause_modifiers

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        if self.clause_modifiers is None:
            self.clause_modifiers = []
        elif not isinstance(self.clause_modifiers, Collection):
            self.clause_modifiers = [self.clause_modifiers]

    # noinspection PyMissingOrEmptyDocstring
    @property
    def theory_evaluator(self) -> TheoryEvaluator:
        return self.learning_system.theory_evaluator

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["learning_system", "theory_metric"]

    @abstractmethod
    def perform_operation(self, targets):
        """
        Applies the operation on the theory, given the target examples.

        :param targets: the target examples
        :type targets: Examples
        :return: the revised theory
        :rtype: NeuralLogProgram or None
        """
        pass

    @abstractmethod
    def theory_revision_accepted(self, revised_theory):
        """
        Method to send a feedback to the revision operator, telling
        that the
        revision was accepted.

        :param revised_theory: the revised theory
        :type revised_theory: NeuralLogProgram
        """
        pass


class BottomClauseBoundedRule(RevisionOperator):
    """
    Operator that implements Guimarães and Paes rule creation algorithm.

    V. Guimarães and A. Paes, Looking at the Bottom and the Top: A Hybrid
    Logical Relational Learning System Based on Answer Sets, 2015 Brazilian
    Conference on Intelligent Systems (BRACIS), Natal, 2015, pp. 240-245.
    """

    def __init__(self,
                 learning_system=None,
                 theory_metric=None,
                 variable_generator=None,
                 relevant_depth=0,
                 refine=False,
                 maximum_side_way_movements=-1,
                 improvement_threshold=0.0,
                 generic=True,
                 evaluation_timeout=300,
                 number_of_process=1):
        super().__init__(learning_system, theory_metric)

        self.variable_generator = variable_generator
        "The variable name generator."

        self.relevant_depth = relevant_depth
        """
        The maximum depth on the transitivity of the relevant concept.
        An atom is relevant to the example if it shares (or transitively shares)
        a term with the example.
        
        If `relevant_depth` is `0`, it means that only the atoms which 
        actually share a term with the example will be considered, 
        these atoms are classed atoms at depth `0`.
        
        If it is `1`, it means that the atoms which share a term with the atoms 
        at depth `0` will also be considered.
        
        If it is `n`, for `n > 0`, it means that the atoms which share a term 
        with the atoms at depth `n - 1` will also be considered.
        
        If it is negative, atoms at any depth will be considered.  
        """

        self.refine = refine
        """
        It specifies if the rule must be reined by adding literals to it,
        in order to try to improve the rule.
        """

        self.maximum_side_way_movements = maximum_side_way_movements
        """
        The maximum side way movements, this is, the maximum number of 
        refining steps will be made, without improving the performance.
        
        If a metric improves by adding a literal to its body, it does not 
        count as a side way movement and the number of side way steps at the 
        moment becomes zero.
        
        If it is negative, there will be no maximum side way movements, 
        wall possible literals will be tried, since it does not degrade the 
        rule.
        """

        self.improvement_threshold = improvement_threshold
        """
        The minimal necessary improvements, over the current clause evaluation 
        and a new candidate, to be considered as improvement. If the threshold
        is not met, it is considered a side way movement.
        
        Use a threshold of `0.0` and a negative `maximum_side_way_movements` 
        to allow the search to test all possible rules.
        
        Use a threshold of `e` and a `maximum_side_way_movements` of `0` to 
        stop as soon as a rule does not improve more than `e`. 
        """

        self.generic = generic
        """
        Flag to specify which rule will be returned in case of a tie in the 
        evaluation of the best rules.
        
        If `generic` is `True`, the most generic tied rule will be returned, 
        this is, the rule whose body has the fewest number of literals in it.
        
        if `generic` is `False`, the most specific rule will be returned, 
        instead, this is, the rule whose body has the most number of literals 
        in it. 
        """

        self.evaluation_timeout = evaluation_timeout
        """
        The maximum amount of time, in seconds, allowed to the evaluation of 
        a rule.
        
        By default, it is 300 seconds, or 5 minutes. 
        """

        self.number_of_process = number_of_process
        """
        The maximum number of process this class is allowed to create in order 
        to concurrently evaluate different rules. 
        
        The default is `1`.
        """

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        if self.variable_generator is None:
            self.variable_generator = DEFAULT_VARIABLE_GENERATOR()
        # noinspection PyAttributeOutsideInit
        self.multiprocessing = MultiprocessingEvaluation(
            self.learning_system, self.theory_metric,
            EquivalentHonClauseAsyncTransformer(),
            self.evaluation_timeout, self.number_of_process
        )

    # noinspection PyMissingOrEmptyDocstring
    def perform_operation(self, targets):
        try:
            logger.info("Performing operation on\t%d examples.", targets.size())
            theory = self.learning_system.theory.copy()
            self.add_null_examples(targets)
            inferred_examples = self.learning_system.infer_examples(targets)
            for example in ExampleIterator(targets):
                self.perform_operation_for_example(
                    example, theory, targets, inferred_examples)
            return theory
        except KnowledgeException as e:
            raise TheoryRevisionException("Error when revising the theory.", e)

    def add_null_examples(self, targets):
        """
        Adds the null example for each predicate in `targets`.

        :param targets: the targets
        :type targets: Examples
        """
        for predicate in targets.keys():
            targets.add_example(
                get_null_example(self.learning_system.knowledge_base,
                                 predicate))

    # noinspection PyMissingOrEmptyDocstring
    def theory_revision_accepted(self, revised_theory):
        pass

    def perform_operation_for_example(self, example, theory, targets,
                                      inferred_examples):
        """
        Performs the operation for a single examples.

        :param example: the example
        :type example: Atom
        :param theory: the theory
        :type theory: NeuralLogProgram
        :param targets: the other examples
        :type targets: Examples
        :param inferred_examples: the inferred value for the examples
        :type inferred_examples: ExamplesInferences
        """
        try:
            if example.weight <= 0.0 or \
                    is_covered(self.learning_system.knowledge_base, example,
                               inferred_examples):
                if example.weight > 0.0:
                    logger.debug("Skipping covered example:\t%s", example)
                # It skips negatives or covered positive examples
                return

            logger.debug("Building clause from the example:\t%s", example)
            bottom_clause = self.build_bottom_clause(example)
            horn_clause = \
                self.build_rule_from_bottom_clause(targets, bottom_clause)
            for clause_modifier in self.clause_modifiers:
                horn_clause = \
                    clause_modifier.modify_clause(horn_clause, targets)
            theory.add_clauses(horn_clause)
            theory.build_program()
            logger.info("Rule appended to the theory:\t%s", horn_clause)
        except (KnowledgeException, InitializationException):
            logger.exception("Error when revising the example, reason:")

    def build_bottom_clause(self, example):
        """
        Builds a bottom clause based on the `example`.

        :param example: the example
        :type example: Atom
        :return: the bottom clause
        :rtype: HornClause
        """
        pass

    def build_rule_from_bottom_clause(self, targets, bottom_clause):
        """
        Builds a Horn clause from a bottom clause, based on the Guimarães and
        Paes rule creation algorithm.

        :param targets: the evaluation targets
        :type targets: Examples
        :param bottom_clause: the bottom clause
        :type bottom_clause: HornClause
        :raise KnowledgeException: in case an error occurs during the revision
        :return: a Horn clause
        :rtype: HornClause
        """
        pass

    # TODO: finish this class
