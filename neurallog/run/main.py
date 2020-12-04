"""
The main Command Line Interface of the system.

It is the main entry point of the system, containing commands for
processing data, training, evaluation and analyses of learned models.
"""

import argparse
import logging
import os
import sys
import time
import traceback

# IMPROVE: import only the command names and description. Then, import only
#  the class of the command the will be used.
import neurallog.run.command
import neurallog.run.command.learn_structure
import neurallog.run.command.output_nlp
import neurallog.run.command.train
import neurallog.run.command.plot_result_it

logger = logging.getLogger(__name__ if __name__ != "__main__" else "main")


def main():
    """
    Runs the system.
    """
    start = time.process_time()
    start_real = time.perf_counter()
    # noinspection PyUnresolvedReferences
    commands = neurallog.run.command.command.all
    program = "python " + os.path.basename(sys.argv[0])
    command_arg_name = "command"
    usage = program + " " + command_arg_name + " [ options... ]\n\n"
    usage += "This class can be used to data processing " + \
             "and to perform experiments.\n\n"
    usage += "The possible commands are:\n\n"
    usage += neurallog.run.command.get_command_docs(commands)
    usage += "\n\n"
    parser = argparse.ArgumentParser(
        usage=usage,
        description="Class for operate a Relational Neural Model.")
    command_help = 'the command to run, the possible commands are: [%s]' \
                   % ", ".join(sorted(commands.keys()))
    parser.add_argument('command', help=command_help)
    args = parser.parse_args(sys.argv[1:2])
    args_command = args.command.lower()
    if args_command not in commands:
        logger.error('Unrecognized command %s', args_command)
        if not neurallog.run.command.suggest_similar_commands(
                args_command, list(commands.keys()), logger=logger):
            parser.print_help()
        logging.shutdown()
        exit(1)
    args = sys.argv[2:]
    start_func = time.process_time()
    start_func_real = time.perf_counter()
    command = commands[args_command](program, args)
    success = 0
    # noinspection PyBroadException
    try:
        command.run()
    except Exception:
        # logger.error("Main program error, reason: %s", e)
        logger.exception("Main program error:")
        traceback.print_exc()
        success = 1

    end = time.process_time()
    end_real = time.perf_counter()
    # logger.info("\n")
    logger.info(
        "The initialisation time of the program was (sys+user / real):    "
        "%0.3fs,\t%0.3fs", start_func - start, start_func_real - start_real)
    logger.info(
        "The       function time of the program was (sys+user / real):    "
        "%0.3fs,\t%0.3fs", end - start_func, end_real - start_func_real)
    logger.info(
        "The          total time of the program was (sys+user / real):    "
        "%0.3fs,\t%0.3fs", end - start, end_real - start_real)
    logging.shutdown()
    exit(success)


if __name__ == "__main__":
    main()
