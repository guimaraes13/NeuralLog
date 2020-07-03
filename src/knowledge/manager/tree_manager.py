"""
Manages the incoming examples manager in a tree representation of the theory.
"""
import collections
import logging
from typing import TypeVar, Generic, Optional, Set, List, Dict

from src.knowledge.examples import Examples, ExampleIterator
from src.knowledge.manager.example_manager import IncomingExampleManager
from src.knowledge.program import NeuralLogProgram
from src.knowledge.theory.manager.revision.revision_examples import \
    RevisionExamples
from src.knowledge.theory.manager.revision.sample_selector import \
    AllRelevantSampleSelect
from src.language.language import HornClause, Atom, Predicate
from src.util.clause_utils import to_variable_atom

logger = logging.getLogger(__name__)

E = TypeVar('E')


class Node(Generic[E]):
    """
    Class to manage a node of the TreeTheory data.
    """

    def __init__(self, parent, element,
                 default_child_element=None, children=None):
        """
        Constructs a node.

        :param parent: the parent of the node
        :type parent: Optional[Node[E]]
        :param element: the element of the node
        :type element: E
        :param default_child_element: the default child element
        :type default_child_element: Optional[E]
        :param children: the children of the node
        :type children: Optional[collections.Iterable[Node[E]]]
        """
        self.parent: Optional[Node] = parent
        self.element: E = element
        self.default_child: Optional[Node[E]] = \
            Node(self, default_child_element)
        self.children: Optional[Set[Node[E]]] = children

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
        return Node(None, element,
                    default_child_element=default_child, children=set())

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
            self, child, default_child_element=default_child, children=set())
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

        if self.parent != other.parent:
            return False

        if self.children != other.children:
            return False

        return self.default_child == other.default_child

    def __repr__(self):
        str(self.element)


class TreeTheory:
    """
    Class to manage the theory as a tree.
    """

    DEFAULT_THEORY_BODY = [NeuralLogProgram.FALSE_ATOM]

    def __init__(self):
        """
        Creates a tree theory.
        """
        self.revision_leaves: Optional[List[Node[HornClause]]] = None
        "The revision leaves"

        self.revision_leaf_index: Optional[int] = None
        "The index of the current revision leaf"

        self.tree_map: Optional[Dict[Predicate, Node[HornClause]]] = None
        self.leaf_examples_map: \
            Optional[Dict[Predicate, Dict[Node[HornClause],
                                          RevisionExamples]]] = None

    def initialize(self, theory):
        """
        Initializes the tree with the theory.

        :param theory: the theory
        :type theory: NeuralLogProgram
        """
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
            root.element.body.append(NeuralLogProgram.TRUE_ATOM)
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
        root = Node.new_tree(TreeTheory.build_default_theory(head),
                             TreeTheory.build_default_theory(head))
        self.tree_map[head.predicate] = root
        return root

    def get_tree_for_example(self, example):
        """
        Retrieves the root of the tree responsible for dealing with the given
        example. If the tree does not exists yet, it is created.

        :param example: the example
        :type example: Atom
        :return: the tree
        :rtype: Node[HornClause]
        """
        root = self.tree_map.get(example.predicate)
        if root is None:
            head = to_variable_atom(example)
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
        return self.leaf_examples_map.get(predicate).pop(leaf, None)

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

    def get_current_revision_leaf(self):
        """
        Gets the current revision leaf.

        :return: the current revision leaf
        :rtype: Node[HornClause]
        """
        if self.revision_leaf_index is None:
            return None
        return self.revision_leaves[self.revision_leaf_index]

    def get_revision_leaf(self, index=None):
        """
        Gets the revision leaf of `index`.

        :param index: the index
        :type index: Optional[int]
        :return: the revision leaf
        :rtype: Node[HornClause]
        """
        if index is None:
            return self.get_current_revision_leaf()
        return self.revision_leaves[index]

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
        return HornClause(head, NeuralLogProgram.FALSE_ATOM)

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
            if len(current_body) == 1 and \
                    NeuralLogProgram.TRUE_ATOM in current_body:
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


class TreeExampleManager(IncomingExampleManager):
    """
    Class to manage the examples by putting them in a three structure based on
    the theory.
    """

    ALL_SAMPLE_SELECTOR = AllRelevantSampleSelect()

    def __init__(self, learning_system=None, sample_selector=None,
                 tree_theory=None):
        super().__init__(learning_system, sample_selector)
        self.tree_theory: TreeTheory = tree_theory
        self.cached_clauses: Dict[Predicate, List[HornClause]] = \
            dict()
        "Caches the list of all rules, except the rules of the key predicate."

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields() + ["tree_theory"]

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
        count = 0
        if not isinstance(examples, collections.Iterable):
            examples = [examples]
        self._update_rule_cache()
        for example in examples:
            self.place_example(modified_leaves, example)
            count += 1
        logger.debug(
            "%s\twew example(s) placed at the leaves of the tree", count)
        return modified_leaves

    def place_example(self, modified_leaves_map, example):
        """
        Places the incoming example into the correct leaves and append the
        leaves in the map of modified leaves.

        :param modified_leaves_map: the map of modified leaves
        :type modified_leaves_map: Dict[Predicate, Set[Node[HornClause]]]
        :param example: the example
        :type example: Atom
        """
        predicate = example.predicate
        modified_leaves = modified_leaves_map.setdefault(predicate, set())
        leaf_examples = \
            self.tree_theory.get_leaf_example_map_from_tree(predicate)
        root = self.tree_theory.get_tree_for_example(example)
        covered = self.transverse_theory_tree(
            root, example, modified_leaves, leaf_examples)
        if not covered:
            self.add_example_to_leaf(
                root.default_child, example, modified_leaves, leaf_examples)

    def transverse_theory_tree(
            self, root, example, modified_leaves, leaf_examples):
        """
        Transverses the theory tree passing the covered example to the
        respective sons and repeating the process of each son. All the leaves
        modified by this process will be appended to `modified_leaves`.

        :param root: the root of the tree
        :type root: Node[HornClause]
        :param example: the example
        :type example: Atom
        :param modified_leaves: the modified leaves set
        :type modified_leaves: Set[Node[HornClause]]
        :param leaf_examples: the leaf examples map to save the examples of
        each lead
        :type leaf_examples: Dict[Node[HornClause], RevisionExamples]
        :return: `True`, if the example is covered; otherwise, `False`
        :rtype: bool
        """
        inferred_examples = self.learning_system.infer_examples(
            example,
            self.cached_clauses.get(example.predicate, []) + [root.element],
            retrain=False)
        covered = inferred_examples.contains_example(example)
        if covered:
            if root.children:
                self.push_example_to_child(
                    root, example, modified_leaves, leaf_examples)
            else:
                self.add_example_to_leaf(
                    root, example, modified_leaves, leaf_examples)

        return covered

    def push_example_to_child(
            self, node, example, modified_leaves, leaf_example):
        """
        Pushes the example to the node, if the node has more children,
        recursively pushes to its children as well.

        :param node: the node
        :type node: Node[HornClause]
        :param example: the example
        :type example: Atom
        :param modified_leaves: the set into which to append the modified
        leaves
        :type modified_leaves: Set[Node[HornClause]]
        :param leaf_example: the map of examples of each leaf, in order to
        append the current example
        :type leaf_example: Dict[Node[HornClause], RevisionExamples]
        """
        covered = False
        for child in node.children:
            covered |= self.transverse_theory_tree(
                child, example, modified_leaves, leaf_example)
        if not covered:
            self.add_example_to_leaf(
                node.default_child, example, modified_leaves, leaf_example)

    def add_example_to_leaf(self, leaf, example, modified_leaves, leaf_example):
        """
        Adds the example to the leaf.

        :param leaf: the leaf
        :type leaf: Node[HornClause]
        :param example: the example
        :type example: Atom
        :param modified_leaves: the set of modified leaves
        :type modified_leaves: Set[Node[HornClause]]
        :param leaf_example: the map of examples per leaf
        :type leaf_example: Dict[Node[HornClause], RevisionExamples]
        # :return: `True`, if this operation changes the set of modified leaves;
        # otherwise, `False`
        # :rtype: bool
        """
        revision_examples = leaf_example.get(leaf)
        if revision_examples is None:
            revision_examples = RevisionExamples(
                self.learning_system, self.sample_selector.copy())
            leaf_example[leaf] = revision_examples
        revision_examples.add_example(example)
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
            targets = []
            for leaf in leaves:
                target = self.tree_theory.get_example_from_leaf(predicate, leaf)
                if target:
                    targets.append(target)
                    self.tree_theory.revision_leaves.append(leaf)
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
