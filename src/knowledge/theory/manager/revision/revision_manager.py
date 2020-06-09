"""
Manages the revision of the theory.
"""
import logging
from abc import abstractmethod

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
    def initialize(self) -> None:
        super().initialize()

    # noinspection PyMissingOrEmptyDocstring,PyMethodMayBeStatic
    def get_required_fields(self):
        return ["theory_revision_manager", "operator_selector"]
