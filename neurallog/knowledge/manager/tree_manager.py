"""
Manages the incoming examples manager in a tree representation of the theory.
"""

import collections
import logging
from typing import TypeVar, Generic, Optional, Set, List, Dict, Tuple

import neurallog.knowledge.theory.manager.revision.revision_examples as re
import neurallog.knowledge.theory.manager.revision.sample_selector as selector
from neurallog.knowledge.examples import Examples, ExampleIterator, \
    ExamplesInferences
from neurallog.knowledge.manager.example_manager import IncomingExampleManager
from neurallog.knowledge.program import NeuralLogProgram
from neurallog.knowledge.theory.manager.revision.clause_modifier import \
    ClauseModifier
from neurallog.language.language import HornClause, Atom, Predicate, Literal, \
    get_variable_atom
from neurallog.util import OrderedSet
from neurallog.util.multiprocessing.evaluation_transformer import \
    apply_modifiers

logger = logging.getLogger(__name__)

E = TypeVar('E')

TRUE_LITERAL = Literal(NeuralLogProgram.TRUE_ATOM)
FALSE_LITERAL = Literal(NeuralLogProgram.FALSE_ATOM)


def group_examples_by_predicate(examples):
    """
    Groups the examples by the predicate.

    :param examples: the examples
    :type examples: collections.Iterable[Atom]
    :return: the grouped examples
    :rtype: Dict[Predicate, Set[Atom]]
    """
    grouped_examples = dict()
    for example in examples:
        grouped_examples.setdefault(
            example.predicate, OrderedSet()).add(example)
    return grouped_examples


def add_clause_to_tree(revised_clause, revision_leaf):
    """
    Adds the revised clause to the tree.

    :param revised_clause: the revised clause
    :type revised_clause: HornClause
    :param revision_leaf: the revised tree node
    :type revision_leaf: Node[HornClause]
    """

    if revision_leaf.is_default_child:
        revision_leaf = revision_leaf.parent
    if revision_leaf.is_root and TreeTheory.is_default_theory(revision_leaf):
        revision_leaf.element.body.clear()
        revision_leaf.element.body.append(TRUE_LITERAL)
        initial_body = []
    else:
        initial_body = revision_leaf.element.body
    TreeTheory.add_nodes_to_tree(revised_clause, revision_leaf, initial_body)


class Node(Generic[E]):
    """
    Class to manage a node of the TreeTheory data.
    """

    def __init__(self, parent, element, default_child_element=None):
        """
        Constructs a node.

        :param parent: the parent of the node
        :type parent: Node[E] or None
        :param element: the element of the node
        :type element: E
        :param default_child_element: the default child element
        :type default_child_element: E or None
        """
        self.parent: Optional[Node] = parent
        self.element: E = element
        if default_child_element is None:
            self.default_child: Optional[Node[E]] = None
            self.children = None
        else:
            self.default_child = Node(self, default_child_element, None)
            self.children: Set[Node[E]] = set()

    @staticmethod
    def new_tree(element, default_child):
        """
        Builds a new tree

        :param element: the element
        :type element: E
        :param default_child: the default child
        :type default_child: E
        :return: the root of the tree
        :rtype: Node[E]
        """
        return Node(None, element, default_child_element=default_child)

    def add_child_to_node(self, child, default_child):
        """
        Adds the child element to this node.

        :param child: the child element
        :type child: E
        :param default_child: the element of the default child of `child`
        :type default_child: E
        :return: the node representation of the child, if succeeds;
        otherwise, `None`
        :rtype: Optional[Node[E]]
        """
        if self.children is None:
            return None

        child_node = Node(
            self, child, default_child_element=default_child)
        self.children.add(child_node)
        return child_node

    def remove_node_from_tree(self):
        """
        Removes this node from the tree.

        :return: `True` if the tree changes; otherwise, `False`
        :rtype: bool
        """
        if self.parent is None:
            return False
        if self not in self.parent.children:
            return False
        self.parent.children.remove(self)
        return True

    @property
    def is_root(self):
        """
        Checks if the node is the root of the tree.

        :return: `True`, if the node is the root of the tree
        :rtype: bool
        """
        return self.parent is None

    @property
    def is_default_child(self):
        """
        Checks if the node is a default child node.

        :return: `True`, if the node is a default child node
        :rtype: bool
        """
        return self.default_child is None

    @property
    def is_leaf(self):
        """
        Checks if the node is a not default leaf.

        :return: `True`, if the node is a not default leaf
        :rtype: bool
        """
        return self.children is not None and len(self.children) == 0

    def __hash__(self):
        result = 31
        result = result * 31 + hash(self.element)
        if self.is_root:
            return result
        result = 31 * hash(self.parent)
        return result

    # noinspection DuplicatedCode
    def __eq__(self, other):
        if id(self) == id(other):
            return True

        if not isinstance(other, Node):
            return False

        if id(self.parent) != id(other.parent):
            return False

        if self.children != other.children:
            return False

        return self.default_child == other.default_child

    def __repr__(self):
        return str(self.element)


class TreeTheory:
    """
    Class to manage the theory as a tree.
    """

    DEFAULT_THEORY_BODY = [FALSE_LITERAL]

    def __init__(self, learning_system=None, initial_modifiers=()):
        """
        Creates a tree theory.

        :param initial_modifiers: Sets the initial clause modifiers. These
        modifiers are used on the creation of the default theories for each
        predicate. Although the clause modifiers accept a set of examples in
        order to modify the clause, these modifiers must assume the examples
        can be `None`.
        :type initial_modifiers: collections.Collection[ClauseModifier]
        """
        self.learning_system = learning_system

        self.revision_leaves: Optional[List[Node[HornClause]]] = None
        "The revision leaves"

        self.current_index: Optional[int] = None
        "The index of the current revision leaf"

        self.target_predicates: Optional[List[Predicate]] = None
        "The target predicates"

        self.tree_map: Optional[Dict[Predicate, Node[HornClause]]] = None
        self.leaf_examples_map: \
            Optional[Dict[Predicate, Dict[Node[HornClause],
                                          re.RevisionExamples]]] = None
        self.initial_modifiers: collections.Collection[ClauseModifier] = \
            initial_modifiers

    def initialize(self, theory=None):
        """
        Initializes the tree with the theory.

        :param theory: the theory
        :type theory: NeuralLogProgram or None
        """
        if not hasattr(self, "initial_modifiers") or \
                self.initial_modifiers is None:
            self.initial_modifiers = ()
        else:
            for clause_modifier in self.initial_modifiers:
                if not hasattr(clause_modifier, "learning_system") or \
                        clause_modifier.learning_system is None:
                    clause_modifier.learning_system = self.learning_system
                clause_modifier.initialize()
        self.revision_leaves = None
        self.target_predicates = None
        self.current_index = None
        self.leaf_examples_map = dict()
        self.tree_map = dict()
        if theory:
            self.build_tree_map(theory.clauses_by_predicate)

    def build_tree_map(self, clauses_by_predicate):
        """
        Builds the tree map of the theory.

        :param clauses_by_predicate: the clauses by predicates
        :type clauses_by_predicate: Dict[Predicate, List[HornClause]]
        """
        for key, value in clauses_by_predicate.items():
            current_clause = value[0]
            root = self.build_initial_tree(current_clause.head)
            root.element.body.clear()
            root.element.body.append(TRUE_LITERAL)
            TreeTheory.add_nodes_to_tree(current_clause, root, [])
            for current_clause in value[1:]:
                parent_node = TreeTheory.find_parent_node(root, current_clause)
                TreeTheory.add_nodes_to_tree(
                    current_clause, parent_node, parent_node.element.body)

    def build_initial_tree(self, head):
        """
        Builds the initial tree for the head.

        :param head: the head
        :type head: Atom
        :return: the root of the initial tree
        :rtype: Node[HornClause]
        """
        initial_clause = TreeTheory.build_default_theory(head)
        # noinspection PyTypeChecker
        initial_clause = \
            apply_modifiers(self.initial_modifiers, initial_clause, None)
        default_clause = TreeTheory.build_default_theory(head)
        # noinspection PyTypeChecker
        default_clause = \
            apply_modifiers(self.initial_modifiers, default_clause, None)
        root = Node.new_tree(initial_clause, default_clause)
        self.tree_map[head.predicate] = root
        return root

    def get_tree_for_example(self, predicate):
        """
        Retrieves the root of the tree responsible for dealing with the given
        predicate. If the tree does not exists yet, it is created.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the tree
        :rtype: Node[HornClause]
        """
        root = self.tree_map.get(predicate)
        if root is None:
            head = get_variable_atom(predicate)
            root = self.build_initial_tree(head)

        return root

    def get_tree_by_predicate(self, predicate):
        """
        Gets the tree by the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the tree
        :rtype: Node[HornClause]
        """
        return self.tree_map.get(predicate)

    def get_leaf_example_map_from_tree(self, predicate):
        """
        Gets the leaf examples map for the predicate. Computes a new one if
        it is absent.

        :param predicate: the predicate
        :type predicate: Predicate
        :return: the leaf examples map
        :rtype: Dict[Node[HornClause], RevisionExamples]
        """
        return self.leaf_examples_map.setdefault(predicate, dict())

    def remove_example_from_leaf(self, predicate, leaf):
        """
        Removes the set of examples from the leaf map of the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :param leaf: the leaf
        :type leaf: Node[HornClause]
        :return: the removed set of examples
        :rtype: RevisionExamples
        """
        return self.leaf_examples_map.get(predicate, dict()).pop(leaf, None)

    def get_example_from_leaf(self, predicate, leaf):
        """
        Gets the examples from the leaf map of the predicate.

        :param predicate: the predicate
        :type predicate: Predicate
        :param leaf: the leaf
        :type leaf: Node[HornClause]
        :return: the set of examples
        :rtype: RevisionExamples
        """
        return self.leaf_examples_map.get(predicate).get(leaf)

    def get_revision_leaf(self, index=None):
        """
        Gets the revision leaf of `index`. If `index` is `None`, returns the
        current revision leaf.

        :param index: the index
        :type index: int or None
        :return: the revision leaf
        :rtype: Node[HornClause]
        """
        if index is None:
            if self.current_index is None:
                return None
            return self.revision_leaves[self.current_index]
        return self.revision_leaves[index]

    def get_target_predicate(self, index=None):
        """
        Gets the target predicate of `index`. If `index` is `None`, returns the
        current target predicate.

        :param index: the index
        :type index: int or None
        :return: the revision leaf
        :rtype: Predicate
        """
        if index is None:
            if self.current_index is None:
                return None
            return self.target_predicates[self.current_index]
        return self.target_predicates[index]

    def remove_literal_from_tree(self, revision_node):
        """
        Removes the literal from the tree and passes it examples to its parent.

        :param revision_node: the literal node to remove
        :type revision_node: Node[HornClause]
        """
        predicate = revision_node.element.head.predicate
        if revision_node.is_leaf:
            # Gets the examples from the last literal, which has been deleted
            examples_from_leaf = self.get_example_from_leaf(
                predicate, revision_node)
            # Unbinds the examples from the leaf that will be removed from
            #  the tree
            self.remove_example_from_leaf(predicate, revision_node)
            # Removes the leaf from the tree
            TreeTheory.remove_node_from_tree(revision_node)
            # Links the examples to the removed leaf's parent
            self.get_leaf_example_map_from_tree(
                predicate)[revision_node.parent] = examples_from_leaf

    def remove_rule_from_tree(self, revision_leaf):
        """
        Removes the rule from the tree and all examples it holds.

        :param revision_leaf: the leaf node of the rule
        :type revision_leaf: Node[HornClause]
        """
        predicate = revision_leaf.element.head.predicate
        grouped_examples = Examples()
        current_leaf: Node[HornClause] = revision_leaf
        while True:
            previous_leaf = current_leaf
            current_leaf = current_leaf.parent
            unproved_examples: Optional[re.RevisionExamples] = \
                self.get_example_from_leaf(
                    predicate, current_leaf.default_child)
            if unproved_examples is not None:
                grouped_examples.add_examples(
                    unproved_examples.get_training_examples(True))
            self.remove_example_from_leaf(predicate, current_leaf.default_child)
            if len(current_leaf.children) != 1 or current_leaf.parent is None:
                break

        self.remove_example_from_leaf(predicate, revision_leaf)
        if current_leaf.is_root and len(current_leaf.children) == 1:
            # Root case
            current_leaf.element.body.clear()
            current_leaf.element.body.append(FALSE_LITERAL)

        if unproved_examples is not None:
            revision_examples = re.RevisionExamples(
                self.learning_system, unproved_examples.sample_selector.copy())
            revision_examples.add_examples(grouped_examples)

            self.get_leaf_example_map_from_tree(predicate)[current_leaf] = \
                revision_examples
        TreeTheory.remove_node_from_tree(previous_leaf)

    @staticmethod
    def is_default_theory(node):
        """
        Checks if the node represents a default theory.

        :param node: the node
        :type node: Node[HornClause]
        :return: `True`, if the node represents a default theory
        :rtype: bool
        """
        if not node.is_root:
            return False

        return node.element.body == TreeTheory.DEFAULT_THEORY_BODY

    @staticmethod
    def build_default_theory(head):
        """
        Builds the default theory for a given head.

        :param head: the head
        :type head: Atom
        :return: the theory
        :rtype: HornClause
        """
        return HornClause(head, FALSE_LITERAL)

    @staticmethod
    def add_node_to_tree(parent, child):
        """
        Adds the child to the parent node.

        :param parent: the parent node
        :type parent: Node[HornClause]
        :param child: the child
        :type child: HornClause
        :return: the child node
        :rtype: Node[HornClause]
        """
        return parent.add_child_to_node(
            child, TreeTheory.build_default_theory(parent.element.head))

    @staticmethod
    def add_nodes_to_tree(clause, revision_leaf, initial_body):
        """
        Adds the nodes from the modified clause to the tree.

        :param clause: the clause
        :type clause: HornClause
        :param revision_leaf: the revised leaf
        :type revision_leaf: Node[HornClause]
        :param initial_body: the initial body
        :type initial_body: List[Literal]
        """
        head = revision_leaf.element.head
        current_body = list(initial_body)
        node = revision_leaf
        for literal in clause.body:
            if literal in current_body:
                continue
            if len(current_body) == 1 and TRUE_LITERAL in current_body:
                current_body.clear()
            next_body = current_body + [literal]
            node = TreeTheory.add_node_to_tree(
                node, HornClause(head, *next_body))
            current_body = next_body

    @staticmethod
    def remove_node_from_tree(node):
        """
        Removes the node from the tree.

        :param node: the node
        :type node: Node[HornClause]
        :return: `True` if the tree changes; otherwise, `False`
        :rtype: bool
        """
        return node.remove_node_from_tree()

    @staticmethod
    def find_parent_node(root, clause):
        """
        Finds the paren node of the clause in the tree.

        :param root: the root of the tree
        :type root: Node[HornClause]
        :param clause: the clause
        :type clause: HornClause
        :return: the parent node
        :rtype: Node[HornClause]
        """
        parent = root
        for literal in clause.body:
            if not parent.children:
                break

            found = False
            for child in parent.children:
                if literal in child.element.body:
                    parent = child
                    found = True
                    break

            if not found:
                break

        return parent

    def has_example(self, example):
        """
        Checks if this tree contains the example in its leaves.

        :param example: the example
        :type example: Atom
        :return: `True` if it has; otherwise, returns `False`
        :rtype: bool
        """
        nodes = self.leaf_examples_map.get(example.predicate, dict())
        for values in nodes.values():
            if values.contains(example):
                return True

        return False


class TreeExampleManager(IncomingExampleManager):
    """
    Class to manage the examples by putting them in a three structure based on
    the theory.
    """

    ALL_SAMPLE_SELECTOR = selector.AllRelevantSampleSelector()

    def __init__(self, learning_system=None, sample_selector=None,
                 tree_theory=None):
        super().__init__(learning_system, sample_selector)
        self.tree_theory: TreeTheory = tree_theory

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields() + ["tree_theory"]

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def initialize(self):
        super().initialize()
        self.cached_clauses: Dict[Predicate, List[HornClause]] = dict()
        "Caches the list of all rules, except the rules of the key predicate."

        self.tree_theory.learning_system = self.learning_system
        self.tree_theory.initialize()

    # noinspection PyMissingOrEmptyDocstring
    def incoming_examples(self, examples):
        modifier_leaves = self.place_incoming_examples(examples)
        self.call_revision(modifier_leaves)

    def _update_rule_cache(self):
        """
        Updates the rule cache.
        """
        self.cached_clauses.clear()
        clauses_by_predicate = self.learning_system.theory.clauses_by_predicate
        for pred1 in clauses_by_predicate.keys():
            for pred2, clauses in clauses_by_predicate.items():
                if pred1 != pred2:
                    self.cached_clauses.setdefault(
                        pred1, []).extend(clauses)

    def place_incoming_examples(self, examples):
        """
        Places the incoming examples into the correct leaves and returns the
        set of the modifier leaves.

        :param examples: the examples
        :type examples: Atom or collections.Iterable[Atom]
        :return: the leaves which was modified due to the addition of examples
        :rtype: Dict[Predicate, Set[Node[HornClause]]]
        """
        modified_leaves: Dict[Predicate, Set[Node[HornClause]]] = dict()
        # count = 0
        self._update_rule_cache()
        if not isinstance(examples, collections.Iterable):
            examples = [examples]
        self.place_examples(modified_leaves, examples)
        # logger.debug(
        #     "%s\t new example(s) placed at the leaves of the tree", count)
        return modified_leaves

    def place_examples(self, modified_leaves_map, examples):
        """
        Places the incoming examples into the correct leaves and append the
        leaves in the map of modified leaves.

        :param modified_leaves_map: the map of modified leaves
        :type modified_leaves_map: Dict[Predicate, Set[Node[HornClause]]]
        :param examples: the examples
        :type examples: collections.Iterable[Atom]
        """
        grouped_examples = Examples()
        grouped_examples.add_examples(examples)
        for predicate, atoms in grouped_examples.items():
            modified_leaves = modified_leaves_map.setdefault(predicate, set())
            leaf_examples = \
                self.tree_theory.get_leaf_example_map_from_tree(predicate)
            root = self.tree_theory.get_tree_for_example(predicate)
            _, not_covered = self.transverse_theory_tree(
                root, Examples({predicate: atoms}),
                modified_leaves, leaf_examples)
            if not_covered:
                self.add_example_to_leaf(root.default_child, not_covered,
                                         modified_leaves, leaf_examples)

    def transverse_theory_tree(
            self, root, examples, modified_leaves, leaf_examples):
        """
        Transverses the theory tree passing the covered example to the
        respective sons and repeating the process of each son. All the leaves
        modified by this process will be appended to `modified_leaves`.

        :param root: the root of the tree
        :type root: Node[HornClause]
        :param examples: the examples
        :type examples: Examples
        :param modified_leaves: the modified leaves set
        :type modified_leaves: Set[Node[HornClause]]
        :param leaf_examples: the leaf examples map to save the examples of
        each lead
        :type leaf_examples: Dict[Node[HornClause], RevisionExamples]
        :return: the set of covered and not examples, respectively
        :rtype: Tuple[Examples, Examples]
        """
        covered = None
        not_covered = None
        # This method is always called for a single predicate
        for predicate in examples.keys():
            if TRUE_LITERAL in root.element.body:
                covered, not_covered = examples, Examples()
            else:
                inferred_examples = self.learning_system.infer_examples(
                    examples,
                    self.cached_clauses.get(predicate, []) + [root.element],
                    retrain=False)
                covered, not_covered = \
                    self.split_covered_examples(examples, inferred_examples)
            if covered:
                if root.children:
                    self.push_example_to_child(
                        root, covered, modified_leaves, leaf_examples)
                else:
                    self.add_example_to_leaf(
                        root, covered, modified_leaves, leaf_examples)

        return covered, not_covered

    @staticmethod
    def split_covered_examples(
            examples, inferences) -> Tuple[Examples, Examples]:
        """
        Splits the examples into the covered and not covered ones.

        :param examples: the examples
        :type examples: Examples
        :param inferences: the inferences
        :type inferences: ExamplesInferences
        :return: the covered and the not covered examples
        :rtype: Tuple[Examples, Examples]
        """
        covered = Examples()
        not_covered = Examples()
        for example in ExampleIterator(examples):
            if inferences.contains_example(example):
                covered.add_example(example)
            else:
                not_covered.add_example(example)

        return covered, not_covered

    def push_example_to_child(
            self, node, examples, modified_leaves, leaf_example):
        """
        Pushes the examples to the node, if the node has more children,
        recursively pushes to its children as well.

        :param node: the node
        :type node: Node[HornClause]
        :param examples: the examples
        :type examples: Examples
        :param modified_leaves: the set into which to append the modified
        leaves
        :type modified_leaves: Set[Node[HornClause]]
        :param leaf_example: the map of examples of each leaf, in order to
        append the current example
        :type leaf_example: Dict[Node[HornClause], RevisionExamples]
        """
        covered_by_children = set()
        for child in node.children:
            covered, _ = self.transverse_theory_tree(
                child, examples, modified_leaves, leaf_example)
            covered_by_children.update(ExampleIterator(covered))
        not_covered = Examples()
        for example in ExampleIterator(examples):
            if example not in covered_by_children:
                not_covered.add_example(example)
        if not_covered:
            self.add_example_to_leaf(
                node.default_child, not_covered, modified_leaves, leaf_example)

    # @staticmethod
    # def examples_not_covered_by_any_child(examples, not_covered_by_child):
    #     """
    #     Returns only the examples that were not covered by any child.
    #
    #     :param examples: all the examples
    #     :type examples: Examples
    #     :param not_covered_by_child: the list of not covered examples of each
    #     child
    #     :type not_covered_by_child: List[Examples]
    #     :return: the examples that were not covered by any child
    #     :rtype: Examples
    #     """
    #     all_not_covered = set(ExampleIterator(not_covered_by_child[0]))
    #     for not_covered in not_covered_by_child[1:]:
    #         all_not_covered.difference_update(set(ExampleIterator(
    #         not_covered)))
    #     not_covered = Examples()
    #     for example in ExampleIterator(examples):
    #         if example in all_not_covered:
    #             not_covered.add_example(example)
    #
    #     return not_covered

    def add_example_to_leaf(
            self, leaf, examples, modified_leaves, leaf_example):
        """
        Adds the example to the leaf.

        :param leaf: the leaf
        :type leaf: Node[HornClause]
        :param examples: the examples
        :type examples: Examples
        :param modified_leaves: the set of modified leaves
        :type modified_leaves: Set[Node[HornClause]]
        :param leaf_example: the map of examples per leaf
        :type leaf_example: Dict[Node[HornClause], RevisionExamples]
        # :return: `True`, if this operation changes the set of modified leaves;
        # otherwise, `False`
        # :rtype: bool
        """
        if not examples:
            return
        revision_examples = leaf_example.get(leaf)
        if revision_examples is None:
            revision_examples = re.RevisionExamples(
                self.learning_system, self.sample_selector.copy())
            leaf_example[leaf] = revision_examples
        added = revision_examples.add_examples(ExampleIterator(examples))
        if added:
            modified_leaves.add(leaf)

    def call_revision(self, modified_leaves):
        """
        Labels each modified leaf as the revision point and call learning
        system for the revision.

        It is not guaranteed that the revision will occur.

        :param modified_leaves: the modified leaves
        :type modified_leaves: Dict[Predicate, Set[Node[HornClause]]]
        """
        for predicate, leaves in modified_leaves.items():
            self.tree_theory.revision_leaves = []
            self.tree_theory.target_predicates = []
            targets = []
            for leaf in leaves:
                target = self.tree_theory.get_example_from_leaf(predicate, leaf)
                if target:
                    targets.append(target)
                    self.tree_theory.revision_leaves.append(leaf)
                    self.tree_theory.target_predicates.append(predicate)
            logger.debug("Calling the revision for\t%d modified leaves of "
                         "predicate:\t%s.", len(targets), predicate)
            self.learning_system.revise_theory(targets)

    # noinspection PyMissingOrEmptyDocstring
    def get_remaining_examples(self):
        examples = Examples()
        for tree in self.tree_theory.leaf_examples_map.values():
            for revision_examples in tree.values():
                training_examples = \
                    revision_examples.get_training_examples(True)
                for atom in ExampleIterator(training_examples):
                    examples.add_example(atom)

        return examples


class RepeatedTreeExampleManager(TreeExampleManager):
    """
    Repeats placing the examples until the tree is no longer modified by it.
    """

    # noinspection PyMissingOrEmptyDocstring
    def incoming_examples(self, examples):
        if isinstance(examples, Atom):
            examples = [examples]
        else:
            examples = list(examples)
        modifier_leaves = True
        count = 0
        while modifier_leaves:
            new_examples = self.filter_existent_examples(examples)
            modifier_leaves = self.place_incoming_examples(new_examples)
            if modifier_leaves:
                count += 1
                logger.info(f"Passing all examples by the {count}-th time.")
                self.call_revision(modifier_leaves)

    def filter_existent_examples(self, examples):
        """
        Filters the examples that are already placed in the tree.

        :param examples: the examples
        :type examples: Atom or collections.Iterable[Atom]
        :return: the examples that are NOT in the tree yet
        :rtype: collections.Iterable[Atom]
        """
        return filter(lambda x: not self.tree_theory.has_example(x), examples)
