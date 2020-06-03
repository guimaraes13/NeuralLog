"""
Handle the evaluation of the revision operators.
"""
from src.util import Initializable


class RevisionOperatorEvaluator(Initializable):
    """
    Class responsible for evaluating the revision operator.
    """

    def __init__(self, revision_operator):
        self.revision_operator = revision_operator
