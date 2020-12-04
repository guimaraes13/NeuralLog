"""
Package to handle the theory.
"""


class TheoryRevisionException(Exception):
    """
    Class to represent an exception during the revision of the theory.
    """

    def __init__(self, message, *args):
        super().__init__(message, *args)
