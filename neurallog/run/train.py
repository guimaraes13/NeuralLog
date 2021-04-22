"""
The main Command Line Interface of the system.

It is the main entry point of the system, containing commands for
processing data, training, evaluation and analyses of learned models.
"""

import logging
import os
import sys
import time

from neurallog.run.command.train import Train

logger = logging.getLogger(__name__)


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
    logging.shutdown()


if __name__ == "__main__":
    main()