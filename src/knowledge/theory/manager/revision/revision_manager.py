"""
Manages the revision of the theory.
"""
import collections
import logging
from abc import ABC, abstractmethod
from typing import Tuple

import src.knowledge.theory.manager.revision.revision_examples as revision
import src.knowledge.theory.manager.revision.revision_operator_selector as ros
import src.knowledge.theory.manager.theory_revision_manager as manager
from src.knowledge.examples import Examples, ExampleIterator
from src.knowledge.theory import TheoryRevisionException
import src.knowledge.theory.manager.revision.operator.revision_operator as ro
from src.util import Initializable

logger = logging.getLogger(__name__)


class RevisionManager(Initializable):
    """
    Class responsible for applying the revision operators to the theory.
    This class applies the revision operators to the theory, in order to
    generate a new theory. This new theory is passed to the
    TheoryRevisionManger, which decides if whether the new theory should
    replace the current theory.
    """

    def __init__(self, theory_revision_manager=None, operator_selector=None):
        """
        Creates a revision manager.

        :param theory_revision_manager: the theory revision manager
        :type theory_revision_manager: manager.TheoryRevisionManager
        :param operator_selector: the operator selector
        :type operator_selector: ros.RevisionOperatorSelector
        """
        self.theory_revision_manager = theory_revision_manager
        self.operator_selector = operator_selector

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        self.operator_selector.learning_system = \
            self.theory_revision_manager.learning_system
        self.operator_selector.initialize()

    # noinspection PyMissingOrEmptyDocstring,PyMethodMayBeStatic
    def required_fields(self):
        return ["theory_revision_manager", "operator_selector"]

    def revise(self, revision_examples):
        """
        Revises the theory based on the revision examples.

        :param revision_examples: the revision examples
        :type revision_examples:
        collections.Collection[revision.RevisionExamples] or
        revision.RevisionExamples
        """
        if not isinstance(revision_examples, collections.Iterable):
            self.call_revision(revision_examples)
        else:
            for revision_example in revision_examples:
                self.call_revision(revision_example)

    def call_revision(self, examples):
        """
        Calls the revision, chosen by the operator selector, on the examples.

        :param examples: the examples
        :type examples: revision.RevisionExamples
        :return: `True`, if the revision was applied; otherwise, `False`
        :rtype: bool
        """
        try:
            return self.theory_revision_manager.apply_revision(
                self.operator_selector, examples)
        except TheoryRevisionException:
            logger.exception("Error when revising the theory, reason:")
        return False


class BestLeafRevisionManager(RevisionManager):
    """
    Class to select the best leaves to revise, based on some heuristic.
    """

    OPTIONAL_FIELDS = RevisionManager.OPTIONAL_FIELDS
    OPTIONAL_FIELDS.update({
        "number_of_leaves_to_revise": -1
    })

    def __init__(self, theory_revision_manager=None, operator_selector=None,
                 tree_theory=None, revision_heuristic=None,
                 number_of_leaves_to_revise=None):
        """
        Creates a revision manager.

        :param theory_revision_manager: the theory revision manager
        :type theory_revision_manager: manager.TheoryRevisionManager
        :param operator_selector: the operator selector
        :type operator_selector: ros.RevisionOperatorSelector
        :param tree_theory: the tree theory
        :type tree_theory: TreeTheory
        :param revision_heuristic: the revision heuristic
        :type revision_heuristic: NodeRevisionHeuristic
        """
        super().__init__(theory_revision_manager, operator_selector)
        self.tree_theory = tree_theory
        self.revision_heuristic = revision_heuristic

        self.number_of_leaves_to_revise = number_of_leaves_to_revise
        if self.number_of_leaves_to_revise is None:
            self.number_of_leaves_to_revise = \
                self.OPTIONAL_FIELDS["number_of_leaves_to_revise"]

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return super().required_fields() + ["tree_theory"]

    # noinspection PyMissingOrEmptyDocstring
    def revise(self, revision_examples):
        if not isinstance(revision_examples, collections.Collection):
            self.tree_theory.revision_leaf_index = 0
            self.call_revision(revision_examples)
        elif len(revision_examples) == 1:
            self.tree_theory.revision_leaf_index = 0
            self.call_revision(next(iter(revision_examples)))
        else:
            total_revision = self.get_maximum_revision_points(revision_examples)
            all_examples = self.theory_revision_manager.train_using_all_examples
            revision_points = []
            index = 0
            for revision_point in revision_examples:
                node = self.tree_theory.get_revision_leaf(index)
                revision_points.append((revision_point, node, index))
                index += 1
            revision_points.sort(
                key=ro.my_cmp_to_key(
                    self.revision_heuristic.compare,
                    lambda x, y, _: (x.get_training_examples(all_examples), y)))
            for revision_point, _, index in revision_points[:total_revision]:
                self.tree_theory.revision_leaf_index = index
                self.call_revision(revision_point)

    def get_maximum_revision_points(self, revision_points):
        """
        Gets the maximum number of revision points that will be used,
        based on the `number_of_leaves_to_revise` and the number of revision
        points.

        :param revision_points: the revision points
        :type revision_points: collections.Collection
            or revision.RevisionExamples
        :return: the number maximum number of revision points
        :rtype: int
        """
        if not isinstance(revision_points, collections.Collection):
            return 1

        length = len(revision_points)
        if self.number_of_leaves_to_revise < 1:
            return length

        return min(self.number_of_leaves_to_revise, length)


class NodeRevisionHeuristic(ABC):
    """
    Class to calculate a heuristic value of how good a collection of examples
    is for revise. The heuristic should be simple and should not rely on
    inference.
    """

    def compare(self, o1, o2):
        """
        Compares two tuples of examples and nodes. By default, as higher the
        heuristic, better the tuple, for revision. Override this method,
        otherwise.

        :param o1: the first tuple of examples and node
        :type o1: Tuple[Examples, Node[HornClause]] or float
        :param o2: the second tuple of examples and node
        :type o2: Tuple[Examples, Node[HornClause]] or float
        :return: `0` if `o1` is equal to `o2`; a value less than `0` if `o1` is
        numerically less than `o2`; a value greater than `0` if `o1` is
        numerically greater than `o2`
        :rtype: float
        """
        if not isinstance(o1, float):
            o1 = self.evaluate(*o1)
        if not isinstance(o2, float):
            o2 = self.evaluate(*o2)
        return o2 - o1

    @abstractmethod
    def evaluate(self, examples, node):
        """
        Evaluates the `node` based on the `examples`.

        :param examples: the examples
        :type examples: Examples
        :param node: the node
        :type node: Node[HornClause]
        :return: the evaluation of the node, based on the examples
        :rtype: float
        """
        pass


class UniformNodeHeuristic(NodeRevisionHeuristic):
    """
    The uniform revision heuristic, treats all nodes as equal.
    """

    # noinspection PyMissingOrEmptyDocstring
    def evaluate(self, examples, node):
        return 0.0


class RepairableNodeHeuristic(NodeRevisionHeuristic):
    """
    Calculates the number of repairable examples in the leaf. The number of
    repairable examples is the number of positive (negative) examples in a
    negative (positive) leaf.
    """

    # noinspection PyMissingOrEmptyDocstring
    def evaluate(self, examples, node):
        if node.is_default_child:
            return sum(map(lambda x: x.weight > 0, ExampleIterator(examples)))
        else:
            return sum(map(lambda x: x.weight <= 0, ExampleIterator(examples)))
