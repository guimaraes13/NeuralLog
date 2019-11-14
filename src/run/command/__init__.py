"""
Package of the possible commands of the system
"""
import logging
from argparse import ArgumentParser
from inspect import getdoc

from utils import SKIP_SIZE, INDENT_SIZE
from utils.data_utils import build_parent_dir

logger = logging.getLogger()

TRAIN_SET_NAME = "train_set"
VALIDATION_SET_NAME = "validation_set"
TEST_SET_NAME = "test_set"

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

        Parameters
        ----------
        name : the name of the command

        Returns
        -------
        out : function
            a function to registry the command with the name
        """

        def registry(cls):
            """
            Registries the command.

            Parameters
            ----------
            cls : Command
                the Command class to be registered.

            Returns
            -------
            out : Command
                the registered command
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


class Command:
    """
    Template class for the commands to appear at the Command Line Interface.
    """

    def __init__(self, program, args):
        """
        Template class for the commands to appear at the Command Line Interface.

        Parameters
        ----------
        program : str
            the program command
        args : list[str]
            the command line arguments to be parsed
        """
        self.program = program
        self.args = args
        self.parser = self.build_parser()
        self.parse_args()

    def build_parser(self) -> ArgumentParser:
        """
        Builds the command line parser.
        Returns
        -------
        out : ArgumentParser
            the command line parser
        """
        pass

    def parse_args(self):
        """
        Parses the command line arguments.
        """

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


def print_args(args):
    """
    Prints the parsed arguments in an organized way.

    Parameters
    ----------
    args : argparse.Namespace or dict
        the parsed arguments
    """
    if isinstance(args, dict):
        arguments = args
    else:
        arguments = args.__dict__
    max_key_length = max(map(lambda x: len(x), arguments.keys()))
    for k, v in sorted(arguments.items()):
        if hasattr(v, "__len__") and len(v) == 1 and not isinstance(v, dict):
            v = v[0]
        logger.info("%s:%s%s", k,
                    " " * (int((max_key_length - len(k)) / SKIP_SIZE)
                           + INDENT_SIZE), v)
    logger.info("")


def create_log_file(log_file):
    """
    Creates a log file.

    Parameters
    ----------
    log_file : str
        the path of the log file
    """
    if log_file is not None:
        build_parent_dir(log_file)
        logging.getLogger().addHandler(
            logging.FileHandler(log_file, mode="w"))
