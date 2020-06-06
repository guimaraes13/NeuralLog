"""
Measures training statistics.
"""


class RunStatistics:
    """
    Class to save statistics about a run of the learning method.
    """

    def __init__(self, knowledge_base_size, examples_size, test_size):
        """
        Creates a run statistic.

        :param knowledge_base_size: the size of the knowledge base
        :type knowledge_base_size: int
        :param examples_size: the number of train examples
        :type examples_size: int
        :param test_size: the number of test examples
        :type test_size: int
        """
        self.knowledge_base_size = knowledge_base_size
        self.examples_size = examples_size
        self.test_size = test_size
        self.train_evaluation = []
        self.test_evaluation = []
