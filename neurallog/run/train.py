#  Copyright 2021 Victor Guimarães
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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
