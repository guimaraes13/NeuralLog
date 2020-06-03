"""
Manages the revision of the theory.
"""
from src.util import Initializable


class RevisionManager(Initializable):
    """
    Class responsible for applying the revision operator on the theory.
    """

    def __init__(self, theory_revision_manager, operator_selector):
        self.theory_revision_manager = theory_revision_manager
        self.operator_selector = operator_selector
