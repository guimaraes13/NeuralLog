"""
Parses the Abstract Syntax Tree.
"""
import logging
from collections import deque

from antlr4 import ParserRuleContext

from language.parser.autogenerated.NeuralLogListener import NeuralLogListener
from language.parser.autogenerated.NeuralLogParser import NeuralLogParser

logger = logging.getLogger()


# class ErrorListener(antlr4.error.ErrorListener.ErrorListener):
#
#     def __init__(self, output):
#         self.output = output
#         self._symbol = ''
#
#     # noinspection PyPep8Naming
#     def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
#         self.output.write(msg)
#         self._symbol = offendingSymbol.text
#
#     @property
#     def symbol(self):
#         return self._symbol


# noinspection PyPep8Naming,PyUnusedLocal,PyUnresolvedReferences,PyRedeclaration
class NeuralLogListenerLogger(NeuralLogListener):
    """
    Parses the Abstract Syntax Tree and logs the elements.
    """

    def __init__(self):
        self.depth = 0

    def log(self, message, context=None, level=logging.DEBUG):
        """
        Logs the current state on the Abstract Syntax Tree.

        :param message: the message to be logged
        :type message: str
        :param context: the current context
        :param level: the level of the log
        :type level: int
        """
        if context is not None:
            logger.log(level, "{}{}:\t{}".format("\t" * self.depth, message,
                                                 context))
        else:
            logger.log(level, "{}{}".format("\t" * self.depth, message))

    def enterProgram(self, ctx: NeuralLogParser.ProgramContext):
        """
        Enter a parse tree produced by NeuralLogParser#program.

        :param ctx: The context
        """
        self.log("Entered program", ctx.getText())
        self.depth += 1

    def exitProgram(self, ctx: NeuralLogParser.ProgramContext):
        """
        Exit a parse tree produced by NeuralLogParser#program.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("Exited program", ctx.getText())

    def enterFor_loop(self, ctx: NeuralLogParser.For_loopContext):
        """
        Enter a parse tree produced by NeuralLogParser#for_loop.

        :param ctx: The context
        """
        self.log("Entered for_loop", ctx.getText())
        self.depth += 1

    def exitFor_loop(self, ctx: NeuralLogParser.For_loopContext):
        """
        Exit a parse tree produced by NeuralLogParser#for_loop.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("Exited for_loop", ctx.getText())

    def enterClause(self, ctx: NeuralLogParser.ClauseContext):
        """
        Enter a parse tree produced by NeuralLogParser#clause.

        :param ctx: The context
        """
        self.log("Entered clause", ctx.getText())
        self.depth += 1

    def exitClause(self, ctx: NeuralLogParser.ClauseContext):
        """
        Exit a parse tree produced by NeuralLogParser#clause.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("Exited clause", ctx.getText())

    def enterAtom(self, ctx: NeuralLogParser.AtomContext):
        """
        Enter a parse tree produced by NeuralLogParser#atom.

        :param ctx: The context
        """
        self.log("Entered atom", ctx.getText())
        self.depth += 1

    def exitAtom(self, ctx: NeuralLogParser.AtomContext):
        """
        Exit a parse tree produced by NeuralLogParser#atom.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("Exited atom", ctx.getText())

    def enterWeighted_atom(self, ctx: NeuralLogParser.Weighted_atomContext):
        """
        Enter a parse tree produced by NeuralLogParser#weighted_atom.

        :param ctx: The context
        """
        self.log("Entered weighted_atom", ctx.getText())
        self.depth += 1

    def exitWeighted_atom(self, ctx: NeuralLogParser.Weighted_atomContext):
        """
        Exit a parse tree produced by NeuralLogParser#weighted_atom.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("Exited weighted_atom", ctx.getText())

    def enterHorn_clause(self, ctx: NeuralLogParser.Horn_clauseContext):
        """
        Enter a parse tree produced by NeuralLogParser#horn_clause.

        :param ctx: The context
        """
        self.log("Entered horn_clause", ctx.getText())
        self.depth += 1

    def exitHorn_clause(self, ctx: NeuralLogParser.Horn_clauseContext):
        """
        Exit a parse tree produced by NeuralLogParser#horn_clause.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("Exited horn_clause", ctx.getText())

    def enterBody(self, ctx: NeuralLogParser.BodyContext):
        """
        Enter a parse tree produced by NeuralLogParser#body.

        :param ctx: The context
        """
        self.log("Entered body", ctx.getText())
        self.depth += 1

    def exitBody(self, ctx: NeuralLogParser.BodyContext):
        """
        Exit a parse tree produced by NeuralLogParser#body.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("Exited body", ctx.getText())

    def enterLiteral(self, ctx: NeuralLogParser.LiteralContext):
        """
        Enter a parse tree produced by NeuralLogParser#literal.

        :param ctx: The context
        """
        self.log("Entered literal", ctx.getText())
        self.depth += 1

    def exitLiteral(self, ctx: NeuralLogParser.LiteralContext):
        """
        Exit a parse tree produced by NeuralLogParser#literal.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("Exited literal", ctx.getText())

    def enterList_of_arguments(self,
                               ctx: NeuralLogParser.List_of_argumentsContext):
        """
        Enter a parse tree produced by NeuralLogParser#list_of_arguments.

        :param ctx: The context
        """
        self.log("Entered list_of_arguments", ctx.getText())
        self.depth += 1

    def exitList_of_arguments(self,
                              ctx: NeuralLogParser.List_of_argumentsContext):
        """
        Exit a parse tree produced by NeuralLogParser#list_of_arguments.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("Exited list_of_arguments", ctx.getText())

    def enterArgument(self, ctx: NeuralLogParser.ArgumentContext):
        """
        Enter a parse tree produced by NeuralLogParser#argument.

        :param ctx: The context
        """
        self.log("Entered argument", ctx.getText())
        self.depth += 1

    def exitArgument(self, ctx: NeuralLogParser.ArgumentContext):
        """
        Exit a parse tree produced by NeuralLogParser#argument.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("Exited arguments", ctx.getText())

    def enterTerm(self, ctx: NeuralLogParser.TermContext):
        """
        Enter a parse tree produced by NeuralLogParser#term.

        :param ctx: The context
        """
        self.log("Entered term", ctx.getText())
        self.depth += 1

    def exitTerm(self, ctx: NeuralLogParser.TermContext):
        """
        Exit a parse tree produced by NeuralLogParser#term.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("Exited term", ctx.getText())

    def enterFor_term(self, ctx: NeuralLogParser.For_termContext):
        """
        Enter a parse tree produced by NeuralLogParser#for_term.

        :param ctx: The context
        """
        self.log("Entered for_term", ctx.getText())
        self.depth += 1

    def exitFor_term(self, ctx: NeuralLogParser.For_termContext):
        """
        Exit a parse tree produced by NeuralLogParser#for_term.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("Exited for_term", ctx.getText())

    def enterNumber(self, ctx: NeuralLogParser.NumberContext):
        """
        Enter a parse tree produced by NeuralLogParser#number.

        :param ctx: The context
        """
        self.log("Entered number", ctx.getText())
        self.depth += 1

    def exitNumber(self, ctx: NeuralLogParser.NumberContext):
        """
        Exit a parse tree produced by NeuralLogParser#number.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("Exited number", ctx.getText())


class NeuralLogKnowledgeExtractor(NeuralLogListenerLogger):
    """
    Extracts the knowledge from the parsed Abstract Syntax Tree.
    """

    def __init__(self):
        super().__init__()
        self.for_stack = deque()

    def enterFor_loop(self, ctx: NeuralLogParser.For_loopContext):
        """
        Enter a parse tree produced by NeuralLogParser#for_loop.

        :param ctx: The context
        """
        self.log("Entered for_loop", ctx.getText(), level=logging.INFO)
        self.for_stack.append(ctx)
        self.depth += 1

    def exitFor_loop(self, ctx: NeuralLogParser.For_loopContext):
        """
        Exit a parse tree produced by NeuralLogParser#for_loop.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("For context:\t{}".format(ctx), level=logging.INFO)
        self.log("Exited for_loop", ctx.getText(), level=logging.INFO)
        # self.for_stack.pop()

    def enterClause(self, ctx: NeuralLogParser.ClauseContext):
        """
        Enter a parse tree produced by NeuralLogParser#clause.

        :param ctx: The context
        """
        self.log("Entered clause", ctx.getText(), level=logging.INFO)
        self.depth += 1

    def exitClause(self, ctx: NeuralLogParser.ClauseContext):
        """
        Exit a parse tree produced by NeuralLogParser#clause.

        :param ctx: The context
        """
        self.depth -= 1
        self.log("Clause context:\t{}".format(ctx), level=logging.INFO)
        self.log("Exited clause", ctx.getText(), level=logging.INFO)


class NeuralLogTransverse:
    """
    Transverse a NeuralLog Abstract Syntax Tree.
    """

    def transverse(self, node):
        """
        Transverse the node from the Abstract Syntax Tree.

        :param node: the node
        :type node: ParserRuleContext
        """
        if isinstance(node, NeuralLogParser.ProgramContext):
            logger.info("Program Node")
            for child in node.children:
                self.transverse(child)
        elif isinstance(node, NeuralLogParser.ClauseContext):
            self.process_clause(node)
            logger.debug("Clause Node")
        elif isinstance(node, NeuralLogParser.For_loopContext):
            self.process_for_loop(node)
            logger.debug("ForLoop Node")

    def process_clause(self, node):
        # TODO: process clause
        if isinstance(node, NeuralLogParser.AtomContext):
            pass
            # self.process_atom(node.get)
        pass

    def process_for_loop(self, node):
        # TODO: process clause
        pass

    def __call__(self, node, *args, **kwargs):
        self.transverse(node)
