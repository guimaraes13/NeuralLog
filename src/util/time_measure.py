"""
Measure elapse time.
"""

import time
from enum import Enum
from typing import Dict, TypeVar, Generic, Callable, List

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


T = TypeVar('T')
R = TypeVar('R')


class TimeMeasure(Generic[T]):
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
        self.timestamps: List[float] = []
        self.stamps_by_name: Dict[T, int] = BiDict()
        self.real_time = real_time
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
        :type name: T
        """
        self.stamps_by_name[name] = len(self.timestamps)
        self.timestamps.append(self.time_function())

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

    def convert_time_measure(self, function: Callable[[T], R]):
        """
        Converts the time measure, based on the function.

        :param function: the conversion function
        :type function: Callable[[T], R]
        :return: the new time measure
        :rtype: TimeMeasure[R]
        """
        time_measure = TimeMeasure(self.real_time)
        time_measure.timestamps.extend(self.timestamps)
        for key, value in self.stamps_by_name.items():
            time_measure.stamps_by_name[function(key)] = value

        return time_measure

    def __repr__(self):
        return "{:.3f}s".format(self.timestamps[-1] - self.timestamps[0])


class RunTimestamps(Enum):
    """
    A enumerator of the names for run timestamps.
    """

    BEGIN = "Begin"
    BEGIN_INITIALIZE = "Begin initializing"
    END_INITIALIZE = "End initializing"
    BEGIN_READ_KNOWLEDGE_BASE = "Begin reading the knowledge base"
    END_READ_KNOWLEDGE_BASE = "End reading the knowledge base"
    BEGIN_READ_THEORY = "Begin reading the theory"
    END_READ_THEORY = "End reading the theory"
    BEGIN_READ_EXAMPLES = "Begin reading the examples"
    END_READ_EXAMPLES = "End reading the examples"
    BEGIN_BUILD_ENGINE_TRANSLATOR = "Begin building the engine translator"
    END_BUILD_ENGINE_TRANSLATOR = "End building the engine translator"
    BEGIN_TRAIN = "Begin training"
    END_TRAIN = "End training"
    BEGIN_EVALUATION = "Begin evaluating"
    END_EVALUATION = "End evaluating"
    BEGIN_DISK_OUTPUT = "Begin disk output"
    END_DISK_OUTPUT = "End disk output"
    END = "Finished"

    def get_message(self):
        """
        Gets the message.

        :return: the message
        :rtype: str
        """

        return self.value


class IterationTimeMessage(Enum):
    """
    Enum to help measuring time of iteration process.
    """

    BEGIN = "Begin of {}"
    LOAD_KNOWLEDGE_DONE = "Load of knowledge from {} done"
    REVISION_DONE = "Revision on {} done"
    TRAIN_EVALUATION_DONE = "Train evaluation on {} done"
    TEST_EVALUATION_DONE = "Test evaluation on {} done"
    EVALUATION_DONE = "End of evaluation on {}"
    SAVING_EVALUATION_DONE = "End of saving evaluation on {}\t to files"
    END = "End of {}"

    def __init__(self, message):
        """
        Creates a iteration time message with `message`.

        :param message: the message
        :type message: str
        """
        self._message = message

    def get_message(self, *args):
        """
        Gets the message.

        :param args: the arguments to format the message
        :type args: Any
        :return: the message
        :rtype: str
        """
        return self._message.format(*args)

    def __eq__(self, other):
        if not isinstance(other, IterationTimeMessage):
            return False

        return self._message == other._message

    def __hash__(self):
        return hash(self._message)


class IterationTimeStamp:
    """
    Class to help measuring time of iteration process.
    """

    def __init__(self, message, iteration_name):
        """
        Creates an iteration time stamp.

        :param message: the message
        :type message: IterationTimeMessage
        :param iteration_name: the iteration name
        :type iteration_name: str
        """
        self.message = message
        self.iteration_name = iteration_name

    def get_message(self):
        """
        Gets the message.

        :return: the message
        :rtype: str
        """
        return self.message.get_message(self.iteration_name)

    def __eq__(self, other):
        if not isinstance(other, IterationTimeStamp):
            return False

        if self.message != other.message:
            return False

        if self.iteration_name != other.iteration_name:
            return False

        return True

    def __hash__(self):
        return hash((self.message, self.iteration_name))

    def __repr__(self):
        return self.get_message()


class IterationTimeStampFactory:
    """
    A factory to create IterationTimeStamps.
    """

    def __init__(self):
        self.messages: \
            Dict[str, Dict[IterationTimeMessage, IterationTimeStamp]] = dict()

    def get_time_stamp(self, message, iteration_name):
        """
        Gets the time stamp of the iteration with the message.

        :param message: the message
        :type message: IterationTimeMessage
        :param iteration_name: the iteration name
        :type iteration_name: str
        :return: the time stamp
        :rtype: IterationTimeStamp
        """

        iteration_message = self.messages.setdefault(iteration_name, dict())

        return iteration_message.setdefault(
            message, IterationTimeStamp(message, iteration_name))
