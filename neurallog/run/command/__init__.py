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
Package of the possible commands of the system
"""
import logging
import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from difflib import SequenceMatcher
from inspect import getdoc

import numpy as np
import scipy.stats

from neurallog.run import configure_log

SKIP_SIZE = 1
INDENT_SIZE = 4

TRAIN_SET_NAME = "train_set"
VALIDATION_SET_NAME = "validation_set"
TEST_SET_NAME = "test_set"


def get_command_docs(commands):
    """
    Gets the documentation of the commands, formatted to a help printing.

    :param commands: the commands
    :type commands: dict
    :return: the formatted documentation for each command
    :rtype: str
    """
    message = "\n"
    max_key_length = max(map(lambda x: len(x), commands.keys()))
    for key in sorted(commands.keys()):
        message += " " * INDENT_SIZE
        message += key
        message += " " * (
                int((max_key_length - len(key)) / SKIP_SIZE) + INDENT_SIZE)
        values = getdoc(commands[key])
        values = values.split("\n") if values else [""]
        message += values[0]
        for value in values[1:]:
            message += "\n"
            message += " " * (int(max_key_length / SKIP_SIZE) + 2 * INDENT_SIZE)
            message += value
        message += "\n\n"
    return message


def build_parent_dir(path):
    """
    Builds the parent directory of the `path`, if it does not exist.

    :param path: the path
    :type path: str
    """
    if path is not None:
        parent_folder = os.path.dirname(path)
        if parent_folder != "" and not os.path.isdir(parent_folder):
            os.makedirs(parent_folder, exist_ok=True)


def suggest_similar_commands(selected_command, possible_commands, logger,
                             similarity_threshold=0.75):
    """
    Suggest the command(s) from the `possible_commands` which is the most
    similar to the `selected_command`.

    :param selected_command: the selected commands
    :type selected_command: str
    :param possible_commands: the possible commands
    :type possible_commands: collections.Iterable[str]
    :param logger: the logger
    :type logger:  logger
    :param similarity_threshold: the smaller similarity accepted to suggest the
    command.
    :type similarity_threshold: float
    :return: True if at least a command is suggested, False otherwise
    :rtype: bool
    """

    def similar(a, b):
        """
        Returns how similar string `a` is from string `b`.

        :param a: string a
        :type a: str
        :param b: string b
        :type b: str
        :return: the similarity between `a` and `b`
        :rtype: float
        """
        return SequenceMatcher(a=a, b=b).ratio()

    options = np.array(possible_commands)
    similarities = [similar(selected_command, x) for x in possible_commands]
    similarities = -np.array(similarities)

    if -similarities.min() < similarity_threshold:
        return False

    rank = scipy.stats.rankdata(similarities, method="dense")
    similar = np.argwhere(rank == 1)
    similar_options = options[similar]
    if similar_options.shape[0] == 1:
        logger.error("Did you mean %s?", similar_options[0, 0])
    else:
        similar_options = options[similar].squeeze()
        logger.error("Did you mean %s or %s?",
                     ", ".join(similar_options[:-1]), similar_options[-1])

    logger.info("")
    return True


def make_command():
    """
    Makes a command from Command class.

    Returns
    -------
    out : function
        a function to registry the command
    """

    commands = {}

    def decorator(name=None):
        """
        Returns a function to registry the command with the name.

        :param name: the name of the command
        :type name: str
        :return: a function to registry the command with the name
        :rtype: function
        """

        def registry(cls):
            """
            Registries the command.

            :param cls: the Command class to be registered.
            :type cls: Command
            :return: the registered command
            :rtype: Command
            """

            func_name = name if name is not None else cls.__name__
            commands[func_name] = cls
            return cls
            # normally a decorator returns a wrapped function,
            # but here we return func unmodified, after registering it

        return registry

    # decorator.all = commands
    setattr(decorator, "all", commands)
    return decorator


command = make_command()


class Command(ABC):
    """
    Template class for the commands to appear at the Command Line Interface.
    """

    def __init__(self, program, args, direct=False):
        """
        Template class for the commands to appear at the Command Line Interface.

        :param program: the program command
        :type program: str
        :param args: the command line arguments to be parsed
        :type args: list[str]
        :param direct: if the command is directly called or if it is under
        another CLI
        :type direct: bool
        """
        self.direct = direct
        self.program = program
        self.args = args
        self.parser = self.build_parser()
        self.parse_args()

    @abstractmethod
    def build_parser(self):
        """
        Builds the command line parser.
        Returns
        -------
        out : ArgumentParser
            the command line parser
        """
        pass

    @abstractmethod
    def parse_args(self):
        """
        Parses the command line arguments.
        """
        configure_log()

    @abstractmethod
    def run(self):
        """
        Runs the command.
        """
        pass

    def get_command_description(self):
        """
        Gets the command description for the help function.

        :return: the command description
        :rtype: str
        """
        return getdoc(self)


def create_log_file(log_file):
    """
    Creates a log file.

    :param log_file: the path of the log file
    :type log_file: str
    """
    if log_file is not None:
        build_parent_dir(log_file)
        handler = logging.FileHandler(log_file, mode="w")
        logger = logging.getLogger()
        handler.setFormatter(logger.handlers[0].formatter)
        logger.addHandler(handler)
