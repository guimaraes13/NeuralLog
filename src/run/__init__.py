"""
Package to manage the run interface.
"""

import logging
import sys

H1 = logging.StreamHandler(sys.stdout)


def configure_log():
    """
    Configures the log handler, message format and log level.
    """
    level = logging.INFO
    # H1.setLevel(level)
    # H1.addFilter(lambda record: record.levelno <= level)
    h2 = logging.StreamHandler(sys.stderr)
    h2.setLevel(logging.WARNING)
    handlers = [H1, h2]
    # noinspection PyArgumentList
    logging.basicConfig(
        format='%(message)s',
        level=level,
        handlers=handlers
    )


configure_log()
