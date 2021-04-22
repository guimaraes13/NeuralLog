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
Package to manage the run interface.
"""

import logging
import sys

DEFAULT_FORMAT = "%(message)s"

H1 = logging.StreamHandler(sys.stdout)
is_log_configured = False


def configure_log(log_format=DEFAULT_FORMAT, level=logging.INFO):
    """
    Configures the log handler, message format and log level.
    """
    global is_log_configured
    if is_log_configured:
        return

    H1.setLevel(level)
    H1.addFilter(lambda record: record.levelno < logging.WARNING)
    h2 = logging.StreamHandler(sys.stderr)
    h2.setLevel(logging.WARNING)
    handlers = [H1, h2]
    # noinspection PyArgumentList
    logging.basicConfig(
        format=log_format,
        level=level,
        handlers=handlers
    )

    is_log_configured = True

# configure_log()
