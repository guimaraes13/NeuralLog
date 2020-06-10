"""
Manages the revision of the theory.
"""
import collections
import logging
from abc import abstractmethod

from src.knowledge.theory import TheoryRevisionException
from src.knowledge.theory.manager.revision.revision_examples import \
    RevisionExamples
from src.knowledge.theory.manager.revision.revision_operator_selector import \
    RevisionOperatorSelector
from src.knowledge.theory.manager.theory_revision_manager import \
    TheoryRevisionManager
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
        :type theory_revision_manager: TheoryRevisionManager
        :param operator_selector: the operator selector
        :type operator_selector: RevisionOperatorSelector
        """
        self.theory_revision_manager = theory_revision_manager
        self.operator_selector = operator_selector

    # noinspection PyMissingOrEmptyDocstring
    @abstractmethod
    def initialize(self):
        super().initialize()
        self.operator_selector.initialize()

    # noinspection PyMissingOrEmptyDocstring,PyMethodMayBeStatic
    def get_required_fields(self):
        return ["theory_revision_manager", "operator_selector"]

    def revise(self, revision_examples):
        """
        Revises the theory based on the revision examples.

        :param revision_examples: the revision examples
        :type revision_examples: collections.Iterable[RevisionExamples]
            or RevisionExamples
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
        :type examples: RevisionExamples
        :return: `True`, if the revision was applied; otherwise, `False`
        :rtype: bool
        """
        try:
            return self.theory_revision_manager.apply_revision(
                self.operator_selector, examples)
        except TheoryRevisionException:
            logger.exception("Error when revising the theory, reason:")
        return False
