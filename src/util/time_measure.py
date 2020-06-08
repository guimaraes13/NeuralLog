"""
Measure elapse time.
"""

import time

from src.knowledge.program import BiDict


def process_time():
    """
    Gets the process time. The difference between two calls of this method
    would result in time that the process has been executed by the processor.

    :return: the process time
    :rtype: float
    """
    return time.process_time()


def performance_time():
    """
    Gets the performance time. The difference between two calls of this method
    would result in the real elapsed time.

    :return: the performance time
    :rtype: float
    """
    return time.perf_counter()


class TimeMeasure:
    """
    A class to measure run time.
    """

    def __init__(self, real_time=True):
        """
        Creates a time measure.

        :param real_time: if `True`, measure the real elapsed time.
        Otherwise, measure the process time.
        :type real_time: bool
        """
        self.timestamps = []
        self.stamps_by_name = BiDict()
        if real_time:
            self.time_function = performance_time
        else:
            self.time_function = process_time

    def time_until_stamp(self, index):
        """
        Returns the time elapsed until the timestamp defined by `index`.

        :param index: the index of the timestamp.
        :type index: int
        :return: the elapsed time
        :rtype: float
        """
        if 0 <= index < len(self.timestamps) or \
                index < 0 and -index <= len(self.timestamps):
            return self.time_function() - self.timestamps[index]

        return 0.0

    def add_measure(self, name):
        """
        Adds a measure of the current timestamp with `name`.

        :param name: the name of the measure/timestamp
        :type name: Any
        """
        self.timestamps.append(self.time_function())
        self.stamps_by_name[name] = len(self.timestamps)

    def time_between_timestamps(self, begin, end):
        """
        Returns the time between `begin` and `end`.

        :param begin: the name of the begin timestamp
        :type begin: Any
        :param end: the name of the end timestamp
        :type end: Any
        :return: the time between the timestamps, in seconds
        :rtype: float
        """
        begin_index = self.stamps_by_name[begin]
        end_index = self.stamps_by_name[end]
        return self.timestamps[end_index] - self.timestamps[begin_index]


