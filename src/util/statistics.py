"""
Measures training statistics.
"""


class RunStatistics:
    """
    Class to save statistics about a run of the learning method.
    """

    def __init__(self, knowledge_base_size=0, train_size=0, test_size=0):
        """
        Creates a run statistic.

        :param knowledge_base_size: the size of the knowledge base
        :type knowledge_base_size: int
        :param train_size: the number of train examples
        :type train_size: int
        :param test_size: the number of test examples
        :type test_size: int
        """
        self.knowledge_base_size = knowledge_base_size
        self.train_size = train_size
        self.test_size = test_size
        self.train_evaluation = []
        self.test_evaluation = []
