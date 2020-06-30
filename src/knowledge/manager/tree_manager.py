"""
Manages the incoming examples manager in a tree representation of the theory.
"""
import logging
from typing import TypeVar, Generic, Optional, Set, List, Dict

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

    def get_revision_leaf(self, index):
        """
        Gets the revision leaf of `index`.

        :param index: the index
        :type index: int
        :return: the revision leaf
        :rtype: Node[HornClause]
        """
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
        self.tree_theory = tree_theory

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields() + ["tree_theory"]

    # noinspection PyMissingOrEmptyDocstring
    def incoming_examples(self, examples):
        pass

    # noinspection PyMissingOrEmptyDocstring
    def get_remaining_examples(self):
        pass
