"""
Handles the revision operators on the TreeTheory.
"""
import logging
from abc import ABC
from typing import Optional, Set

from src.knowledge.examples import Examples
from src.knowledge.manager.tree_manager import TreeTheory, Node, \
    FALSE_LITERAL, TRUE_LITERAL
from src.knowledge.program import NeuralLogProgram
from src.knowledge.theory.manager.revision.operator.literal_appender_operator \
    import \
    LiteralAppendOperator
from src.knowledge.theory.manager.revision.operator.revision_operator import \
    RevisionOperator
from src.language.language import HornClause, Literal
from src.util import OrderedSet
from src.util.multiprocessing.theory_evaluation import AsyncTheoryEvaluator, \
    SyncTheoryEvaluator

logger = logging.getLogger(__name__)


class TreeRevisionOperator(RevisionOperator, ABC):
    """
    Super class for revision operator that performs operation in the TreeTheory.
    """

    def __init__(self, learning_system=None, theory_metric=None,
                 clause_modifiers=None, tree_theory=None):
        """
        Creates a tree revision operator.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param theory_metric: the theory metric
        :type theory_metric: TheoryMetric
        :param clause_modifiers: a clause modifier, a list of clause modifiers
        or none
        :type clause_modifiers: ClauseModifier or Collection[ClauseModifier]
        or None
        :param tree_theory: the tree theory
        :type tree_theory: TreeTheory
        """
        super().__init__(learning_system, theory_metric, clause_modifiers)
        self.tree_theory = tree_theory

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields() + ["tree_theory"]


class AddNodeTreeRevisionOperator(TreeRevisionOperator):
    """
    Revision operator that adds new nodes on the tree theory.

    The nodes to be added will depend on the implementation of the append
    operator.
    """

    OPTIONAL_FIELDS = TreeRevisionOperator.OPTIONAL_FIELDS
    OPTIONAL_FIELDS.update({
        "refine": False,
        "maximum_side_way_movements": -1,
        "improvement_threshold": 0.0,
        "generic": True
    })

    def __init__(self, learning_system=None, theory_metric=None,
                 clause_modifiers=None, tree_theory=None,
                 append_operator=None, refine=None,
                 maximum_side_way_movements=None, improvement_threshold=None,
                 generic=None):
        super().__init__(
            learning_system, theory_metric, clause_modifiers, tree_theory)
        self.append_operator: Optional[LiteralAppendOperator] = append_operator
        self.revised_clause: Optional[HornClause] = None

        self.refine = refine
        "Flag to specify if the rule must be refined or not."

        if self.refine is None:
            self.refine = self.OPTIONAL_FIELDS["refine"]

        self.maximum_side_way_movements = maximum_side_way_movements
        """
        Represents the maximum side way movements, i.e. the number of literals 
        that will be added to the body of the Horn clause without improving the 
        metric.
        
        If the metric improves by adding a literal to the body of the clause, 
        it does not count as a side way movement.
        
        If it is negative, it means that there will be no maximum side way 
        movements, it will be limited by the number of possible literals to be 
        added.
        """

        if self.maximum_side_way_movements is None:
            self.maximum_side_way_movements = \
                self.OPTIONAL_FIELDS["maximum_side_way_movements"]

        self.improvement_threshold = improvement_threshold
        """
        The minimal necessary difference, between the current Horn clause and 
        a new candidate clause, to be considered as an improvement. If the 
        threshold is not met, it is considered a side way movement.
        
        Use a threshold of `0.0` and a negative maximum side way movement to 
        allow the search to test all possible clauses. This might take a long 
        time to run.
        
        Use a threshold of `e` and a maximum side way movement of `0` to stop 
        as soon as a Horn clause does not improve, more than `e`, over the 
        current clause.
        """

        if self.improvement_threshold is None:
            self.improvement_threshold = \
                self.OPTIONAL_FIELDS["improvement_threshold"]

        self.generic = generic
        """
        Flag to specify which Horn clause will be returned in case of a tie 
        in the evaluation metric.
        
        If `generic` is `True`, the most GENERIC clause will be returned, 
        i.e. the one whose body is the smallest.
        
        If `generic` is `False`, the most SPECIFIC clause will be returned, 
        i.e. the one whose body is the largest.
        """

        if self.generic is None:
            self.generic = self.OPTIONAL_FIELDS["generic"]

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields() + ["append_operator"]

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        self.append_operator.learning_system = self.learning_system
        self.append_operator.initialize()

    # noinspection PyMissingOrEmptyDocstring
    def perform_operation(self, targets):
        revision_leaf = self.tree_theory.get_revision_leaf()
        logger.debug("Trying to refine rule:\t%s", revision_leaf)
        if revision_leaf.is_root:
            # This is the root node
            return self.add_rule_to_theory(revision_leaf, targets)
        elif revision_leaf.is_default_child:
            # This node represents a false leaf, it is a rule creation in the
            # parent node
            return self.add_rule_to_theory(revision_leaf.parent, targets)
        else:
            # This node represents a rule, it is a literal addition operation
            return self.add_literal_to_theory(revision_leaf, targets)

    def add_rule_to_theory(self, node, examples):
        """
        Adds a rule, represented by the node with the addition of a set of
        literals, to the theory.

        :param node: the node
        :type node: Node[HornClause]
        :param examples: the examples
        :type examples: Examples
        :return: the modified theory
        :rtype: NeuralLogProgram
        """
        return self.create_sorted_theory(node, examples, False)

    def add_literal_to_theory(self, node, examples):
        """
        Adds a set of literals to the rule, represented by the node.

        :param node: the node
        :type node: Node[HornClause]
        :param examples: the examples
        :type examples: Examples
        :return: the modified theory
        :rtype: NeuralLogProgram
        """
        return self.create_sorted_theory(node, examples, True)

    def create_sorted_theory(self, node, examples, remove_old):
        """
        Creates a sorted theory and adds a new rule from the creation of an
        append of literals from the examples, using the literal append operator.

        :param node: the node
        :type node: Node[HornClause]
        :param examples: the examples
        :type examples: Examples
        :param remove_old: if it is to remove the initial rule, defined by
        the node, from the theory
        :type remove_old: bool
        :return: the new sorted theory
        :rtype: NeuralLogProgram
        """
        element = \
            HornClause(node.element.head) if node.is_root else node.element
        horn_clause = self.append_operator.build_extended_horn_clause(
            examples, element, self.build_redundant_literals(node))
        if not horn_clause:
            return None
        if self.refine:
            horn_clause = self.refine_clause(horn_clause, examples)
        self.revised_clause = horn_clause.horn_clause
        self.revised_clause = \
            self.apply_clause_modifiers(self.revised_clause, examples)
        self.log_changes(node, remove_old)

        theory = self.learning_system.theory.copy()
        theory.add_clauses(self.revised_clause)
        if remove_old:
            self.remove_old_rule_from_theory(node, theory)

        clauses = []
        for predicate_clauses in theory.clauses_by_predicate.values():
            clauses.extend(predicate_clauses)
        clauses.sort(key=lambda x: str(x))

        sorted_theory = NeuralLogProgram()
        sorted_theory.add_clauses(clauses)

        return sorted_theory

    def refine_clause(self, initial_clause, examples):
        """
        Refines the clause. It starts from the `initial_clause` and adds a
        set of literals at a time into its body, based on the append operator.
        At each time, it gets the best possible Horn clause. It finished when
        one of the following criteria is met:

        1) the addition of another set does not improve the Horn clause in
        `maximum_side_way_movements` times;

        2) there is no more possible additions to make.

        After it finishes, it returns the best Horn clause found, based on
        the `generic` parameter.

        :param initial_clause: the initial candidate clause
        :type initial_clause: AsyncTheoryEvaluator or SyncTheoryEvaluator
        :param examples: the examples
        :type examples: Examples
        :return: the best Horn clause found
        :rtype: AsyncTheoryEvaluator or SyncTheoryEvaluator
        """
        logger.debug("Refining rule:\t%s", initial_clause)
        side_way_movements = 0
        best_clause = initial_clause
        current_clause = initial_clause
        while not self.is_to_stop_by_side_way_movements(side_way_movements):
            current_clause = self.append_operator.build_extended_horn_clause(
                examples, current_clause.horn_clause, set())
            if not current_clause:
                break
            logger.debug("Proposed refined rule:\t%s", current_clause)
            improvement = self.theory_metric.difference(
                current_clause.evaluation, best_clause.evaluation)
            if improvement > self.improvement_threshold:
                best_clause = current_clause
                side_way_movements = 0
            else:
                side_way_movements += 1
                if improvement >= 0.0 and not self.generic:
                    best_clause = current_clause
        return best_clause

    def log_changes(self, node, remove_old):
        """
        Logs the changes to the theory.

        :param node: the node of the change
        :type node: Node[HornClause]
        :param remove_old: if it is to remove the old clause
        :type remove_old: bool
        """
        if logger.isEnabledFor(logging.DEBUG):
            if remove_old:
                body = set(self.revised_clause.body)
                body = body.difference(node.element.body)
                logger.debug("Propose to add the literal(s):\t%s", body)
            else:
                logger.debug(
                    "Propose to add the rule:\t%s", self.revised_clause)

    def is_to_stop_by_side_way_movements(self, side_way_movements):
        """
        Checks if it is to stop due to reaching the maximum side way movements.

        :param side_way_movements: the number of iterations without improvement
        :type side_way_movements: int
        :return: `True`, if it is to stop due to the maximum side way
        movements; otherwise, `False`
        :rtype: bool
        """
        return 0 <= self.maximum_side_way_movements < side_way_movements

    # noinspection PyMissingOrEmptyDocstring
    def theory_revision_accepted(self, revised_theory):
        revision_leaf = self.tree_theory.get_revision_leaf()
        self.tree_theory.remove_example_from_leaf(
            revision_leaf.element.head.predicate, revision_leaf)
        if revision_leaf.is_default_child:
            revision_leaf = revision_leaf.parent
        if revision_leaf.is_root and \
                TreeTheory.is_default_theory(revision_leaf):
            revision_leaf.element.body.clear()
            revision_leaf.element.body.append(TRUE_LITERAL)
            initial_body = []
        else:
            initial_body = revision_leaf.element.body
        TreeTheory.add_nodes_to_tree(
            self.revised_clause, revision_leaf, initial_body)

    @staticmethod
    def build_redundant_literals(node):
        """
        Gets the last literal of the body of each child of the node. This
        allows the to create another rule avoiding creating already existing
        rules.

        :param node: the node
        :type node: Node[HornClause]
        :return: the redundant literals
        :rtype: Set[Literal]
        """
        redundant_literals: Set[Literal] = set()
        for sibling in node.children:
            if sibling.element.body:
                redundant_literals.add(sibling.element.body[-1])

        return redundant_literals

    @staticmethod
    def remove_old_rule_from_theory(node, theory):
        """
        Removes the rule from the theory.

        :param node: the node containing the rule
        :type node: Node[HornClause]
        :param theory: the theory
        :type theory: NeuralLogProgram
        """
        theory.clauses_by_predicate[node.element.head.predicate].remove(
            node.element)

    def __repr__(self):
        return f"[{self.__class__.__name__}] {self.append_operator}"


class RemoveNodeTreeRevisionOperator(TreeRevisionOperator):
    """
    Revision operator that removes a node from the tree theory.
    """

    # noinspection PyMissingOrEmptyDocstring
    def perform_operation(self, targets):
        revision_leaf = self.tree_theory.get_revision_leaf()
        logger.debug("Trying to refine rule:\t%s", revision_leaf)
        if revision_leaf.is_root:
            # Root case
            # This is the root node
            if TreeTheory.is_default_theory(revision_leaf):
                # It is the default theory, there is nothing to remove here
                return None
            else:
                # It is the true theory, making it the false (default) theory
                return self.remove_rule_from_theory(revision_leaf)
        elif revision_leaf.is_default_child:
            if len(revision_leaf.parent.children) == 1:
                # Remove Literal case
                revision_node = next(iter(revision_leaf.parent.children))
                if revision_node.is_leaf:
                    # This node represents the last default of a straight rule
                    #  path, it is a literal deletion operation
                    return self.remove_literal_from_theory(
                        revision_node, targets)
        elif len(revision_leaf.parent.children) > 1:
            # Remove Rule case
            # This node represents a bifurcation of a rule, it is a rule
            #  deletion operation
            return self.remove_rule_from_theory(revision_leaf)

        return None

    def remove_rule_from_theory(self, node):
        """
        Removes the rule, represented by the `node`, from the theory.

        :param node: the node
        :type node: Node[HornClause]
        :return: the modified theory
        :rtype: NeuralLogProgram
        """
        theory = self.learning_system.theory.copy()
        element = node.element
        theory.clauses_by_predicate[element.head.predicate].remove(element)
        logger.debug("Propose to remove the rule:\t%s", element)
        return theory

    def remove_literal_from_theory(self, node, examples):
        """
        Removes the literal from the rule, represented by the `node`,
        reducing it to its parent.

        :param node: the node
        :type node: Node[HornClause]
        :param examples: the examples
        :type examples: Examples
        :return: the modified theory
        :rtype: NeuralLogProgram
        """
        predicate = node.element.head.predicate
        theory_clauses = self.learning_system.theory.clauses_by_predicate.get(
            predicate, ())
        clauses = OrderedSet()
        for clause in theory_clauses:
            if node.element == clause:
                revised_clause = \
                    self.apply_clause_modifiers(node.parent.element, examples)
                clauses.add(revised_clause)
            else:
                clauses.add(clauses)
        if logger.isEnabledFor(logging.DEBUG):
            body = set(node.element.body)
            body.difference(node.parent.element.body)
            logger.debug("Propose to remove the literal:\t%s", body)

        modified_theory = self.learning_system.theory.copy()
        modified_theory.clauses_by_predicate[predicate] = list(clauses)
        modified_theory.build_program()
        return modified_theory

    # noinspection PyMissingOrEmptyDocstring
    def theory_revision_accepted(self, revised_theory):
        revision_leaf = self.tree_theory.get_revision_leaf()
        predicate = revision_leaf.element.head.predicate
        self.tree_theory.remove_example_from_leaf(predicate, revision_leaf)
        if revision_leaf.is_root:
            # Root case
            revision_leaf.element.body.clear()
            revision_leaf.element.body.append(FALSE_LITERAL)
        elif revision_leaf.is_default_child and \
                len(revision_leaf.parent.children) == 1:
            # Remove Literal case
            revision_node = next(iter(revision_leaf.parent.children))
            self.remove_literal_from_tree(revision_node)
        else:
            # Remove Rule case
            TreeTheory.remove_node_from_tree(revision_leaf)
            self.tree_theory.remove_example_from_leaf(predicate, revision_leaf)

    def remove_literal_from_tree(self, revision_node):
        """
        Removes the literal from the tree and passes it examples to its parent.

        :param revision_node: the literal node to remove
        :type revision_node: Node[HornClause]
        """
        predicate = revision_node.element.head.predicate
        if revision_node.is_leaf:
            # Gets the examples from the last literal, which has been deleted
            examples_from_leaf = self.tree_theory.get_example_from_leaf(
                predicate, revision_node)
            # Unbinds the examples from the leaf that will be removed from
            #  the tree
            self.tree_theory.remove_example_from_leaf(predicate, revision_node)
            # Removes the leaf from the tree
            TreeTheory.remove_node_from_tree(revision_node)
            # Links the examples to the removed leaf's parent
            self.tree_theory.get_leaf_example_map_from_tree(
                predicate)[revision_node.parent] = examples_from_leaf
