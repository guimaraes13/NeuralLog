"""
The main Command Line Interface of the system.

It is the main entry point of the system, containing commands for
processing data, training, evaluation and analyses of learned models.
"""

import logging
import os
import sys

# noinspection DuplicatedCode
logger = logging.getLogger()


# noinspection DuplicatedCode
def configure_log():
    """
    Configures the log handler, message format and log level.
    """
    level = logging.INFO
    h1 = logging.StreamHandler(sys.stdout)
    h1.setLevel(level)
    # h1.addFilter(lambda record: record.levelno <= level)
    h2 = logging.StreamHandler(sys.stderr)
    h2.setLevel(logging.WARNING)
    handlers = [h1, h2]
    # noinspection PyArgumentList
    logging.basicConfig(
        format='%(message)s',
        level=level,
        handlers=handlers
    )


configure_log()

import time
from src.run.command.train import Train


def main():
    """
    Runs the system.
    """
    start = time.process_time()
    start_real = time.perf_counter()

    program = "python " + os.path.basename(sys.argv[0])
    command = Train(program, sys.argv[1:], direct=True)
    command.run()
    end = time.process_time()
    end_real = time.perf_counter()
    logger.info("\n")
    logger.info("The total time of the program was (sys+user / real):\t"
                "%0.3fs,\t%0.3fs", end - start, end_real - start_real)


if __name__ == "__main__":
    main()
