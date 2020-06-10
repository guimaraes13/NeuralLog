"""
Package to handle the theory.
"""


class TheoryRevisionException(Exception):
    """
    Class to represent an exception during the revision of the theory.
    """

    def __init__(self, message):
        super().__init__(message)
