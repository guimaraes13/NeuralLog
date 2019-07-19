# Generated from NeuralLog.g4 by ANTLR 4.7.2
from antlr4 import *

if __name__ is not None and "." in __name__:
    from .NeuralLogParser import NeuralLogParser
else:
    from NeuralLogParser import NeuralLogParser


# This class defines a complete listener for a parse tree produced by NeuralLogParser.
class NeuralLogListener(ParseTreeListener):

    # Enter a parse tree produced by NeuralLogParser#program.
    def enterProgram(self, ctx: NeuralLogParser.ProgramContext):
        pass

    # Exit a parse tree produced by NeuralLogParser#program.
    def exitProgram(self, ctx: NeuralLogParser.ProgramContext):
        pass

    # Enter a parse tree produced by NeuralLogParser#for_loop.
    def enterFor_loop(self, ctx: NeuralLogParser.For_loopContext):
        pass

    # Exit a parse tree produced by NeuralLogParser#for_loop.
    def exitFor_loop(self, ctx: NeuralLogParser.For_loopContext):
        pass

    # Enter a parse tree produced by NeuralLogParser#clause.
    def enterClause(self, ctx: NeuralLogParser.ClauseContext):
        pass

    # Exit a parse tree produced by NeuralLogParser#clause.
    def exitClause(self, ctx: NeuralLogParser.ClauseContext):
        pass

    # Enter a parse tree produced by NeuralLogParser#atom.
    def enterAtom(self, ctx: NeuralLogParser.AtomContext):
        pass

    # Exit a parse tree produced by NeuralLogParser#atom.
    def exitAtom(self, ctx: NeuralLogParser.AtomContext):
        pass

    # Enter a parse tree produced by NeuralLogParser#weighted_atom.
    def enterWeighted_atom(self, ctx: NeuralLogParser.Weighted_atomContext):
        pass

    # Exit a parse tree produced by NeuralLogParser#weighted_atom.
    def exitWeighted_atom(self, ctx: NeuralLogParser.Weighted_atomContext):
        pass

    # Enter a parse tree produced by NeuralLogParser#horn_clause.
    def enterHorn_clause(self, ctx: NeuralLogParser.Horn_clauseContext):
        pass

    # Exit a parse tree produced by NeuralLogParser#horn_clause.
    def exitHorn_clause(self, ctx: NeuralLogParser.Horn_clauseContext):
        pass

    # Enter a parse tree produced by NeuralLogParser#body.
    def enterBody(self, ctx: NeuralLogParser.BodyContext):
        pass

    # Exit a parse tree produced by NeuralLogParser#body.
    def exitBody(self, ctx: NeuralLogParser.BodyContext):
        pass

    # Enter a parse tree produced by NeuralLogParser#literal.
    def enterLiteral(self, ctx: NeuralLogParser.LiteralContext):
        pass

    # Exit a parse tree produced by NeuralLogParser#literal.
    def exitLiteral(self, ctx: NeuralLogParser.LiteralContext):
        pass

    # Enter a parse tree produced by NeuralLogParser#list_of_arguments.
    def enterList_of_arguments(self,
                               ctx: NeuralLogParser.List_of_argumentsContext):
        pass

    # Exit a parse tree produced by NeuralLogParser#list_of_arguments.
    def exitList_of_arguments(self,
                              ctx: NeuralLogParser.List_of_argumentsContext):
        pass

    # Enter a parse tree produced by NeuralLogParser#argument.
    def enterArgument(self, ctx: NeuralLogParser.ArgumentContext):
        pass

    # Exit a parse tree produced by NeuralLogParser#argument.
    def exitArgument(self, ctx: NeuralLogParser.ArgumentContext):
        pass

    # Enter a parse tree produced by NeuralLogParser#term.
    def enterTerm(self, ctx: NeuralLogParser.TermContext):
        pass

    # Exit a parse tree produced by NeuralLogParser#term.
    def exitTerm(self, ctx: NeuralLogParser.TermContext):
        pass

    # Enter a parse tree produced by NeuralLogParser#for_term.
    def enterFor_term(self, ctx: NeuralLogParser.For_termContext):
        pass

    # Exit a parse tree produced by NeuralLogParser#for_term.
    def exitFor_term(self, ctx: NeuralLogParser.For_termContext):
        pass

    # Enter a parse tree produced by NeuralLogParser#number.
    def enterNumber(self, ctx: NeuralLogParser.NumberContext):
        pass

    # Exit a parse tree produced by NeuralLogParser#number.
    def exitNumber(self, ctx: NeuralLogParser.NumberContext):
        pass
