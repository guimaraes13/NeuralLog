"""
Module to parse the NeuralLog Programs using the ply parser library.
"""

import logging
import sys
from collections import deque
from enum import IntEnum
from typing import List

import ply.lex as lex
import ply.yacc as yacc

from src.language.language import Atom, Predicate, TemplatePredicate, Number, \
    Quote, TemplateTerm, Variable, Constant, Literal, HornClause, AtomClause, \
    BadArgumentException, Clause, FileDefinedClause, ListTerms
from src.language.parser.neural_log_listener import KeyDict, \
    BadClauseException, PLACE_HOLDER, ground_placeholders, solve_place_holders

logger = logging.getLogger(__name__)


def config_log(level=logging.DEBUG):
    """
    Configs the log.
    """
    h1 = logging.StreamHandler(sys.stdout)
    h1.setLevel(level)
    h1.addFilter(lambda record: record.levelno <= logging.WARNING)
    h2 = logging.StreamHandler(sys.stderr)
    h2.setLevel(logging.WARNING)
    handlers = [h1, h2]
    # handlers = [h1]
    # noinspection PyArgumentList
    logging.basicConfig(
        format='%(message)s',
        level=level,
        handlers=handlers
    )


class ContainerTypes(IntEnum):
    """
    Enumerates the types of containers.
    """

    PLACE_HOLDER_CONTAINER = 0
    LIST_OF_ARGUMENTS = 1

    def __repr__(self):
        return self.name


class ListContainer:
    """
    Holds a list of a type.
    """

    def __init__(self, container_type: ContainerTypes, items: List):
        self.container_type = container_type
        self.items = items

    def __getitem__(self, item):
        return self.items[item]

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f"{self.container_type.name}:\t{self.items}"


# noinspection PyPep8Naming,PyMethodMayBeStatic,PySingleQuotedDocstring
class NeuralLogLexer:
    """
    Lexer for the NeuralLog Language.
    """
    # List of reserved words.
    reserved = {
        'for': "FOR_LOOP",
        'in': "IN_TOKEN",
        'do': "DO_TOKEN",
        'done': "DONE_TOKEN",
        'not': "NEGATION",
    }

    # List of token names.
    tokens = [
                 "SCIENTIFIC_NUMBER",
                 "DECIMAL",
                 "INTEGER",
                 "TERM",
                 "PLACE_HOLDER",
                 "OPEN_ARGUMENTS",
                 "CLOSE_ARGUMENTS",
                 "OPEN_LIST_ARGUMENT",
                 "CLOSE_LIST_ARGUMENT",
                 "OPEN_CURLY_BRACES",
                 "CLOSE_CURLY_BRACES",
                 "RANGER_SEPARATOR",
                 "ITEM_SEPARATOR",
                 "END_OF_CLAUSE",
                 "WEIGHT_SEPARATOR",
                 "IMPLICATION_SIGN",
                 "QUOTED",
                 "COMMENT",
                 "BLOCK_COMMENT"
             ] + list(reserved.values())

    t_PLACE_HOLDER = r"{([a-zA-Z0-9_-])+}"
    t_OPEN_ARGUMENTS = r"\("
    t_CLOSE_ARGUMENTS = r"\)"
    t_OPEN_LIST_ARGUMENT = r"\["
    t_CLOSE_LIST_ARGUMENT = r"\]"
    t_OPEN_CURLY_BRACES = r"\{"
    t_CLOSE_CURLY_BRACES = r"\}"
    t_RANGER_SEPARATOR = r"\.\."
    t_ITEM_SEPARATOR = r","
    t_END_OF_CLAUSE = r"\."
    t_WEIGHT_SEPARATOR = r"::"
    t_IMPLICATION_SIGN = r":-"
    t_QUOTED = r"(\"(\\.|[^\"])*\"|\'(\\.|[^\'])*\')"

    def __init__(self, **kwargs):
        """
        Creates a NeuralLog lexer.

        :param kwargs: optional arguments to be passed to the ply lexer
        library.
        :type kwargs: dict
        """
        self.lexer = lex.lex(module=self, **kwargs)

    def t_SCIENTIFIC_NUMBER(self, t):
        r"-?[0-9]*\.[0-9]+[eE][+-]?[0-9]+"
        t.value = float(t.value)
        return t

    def t_DECIMAL(self, t):
        r"-?[0-9]*\.[0-9]+"
        t.value = float(t.value)
        return t

    def t_INTEGER(self, t):
        r"-?[0-9]+"
        t.value = int(t.value)
        return t

    def t_TERM(self, t):
        r"[a-zA-Z_-][a-zA-Z0-9_-]*"
        t.type = self.reserved.get(t.value, "TERM")  # Check for reserved words
        return t

    def t_COMMENT(self, t):
        r'(\#|%)[^\r\n]*'
        pass

    def t_BLOCK_COMMENT(self, t):
        r"/\*([^\*]|\*[^/])*\*/"
        # r"/\*.*\*/"
        t.lexer.lineno += len(t.value.split("\n"))

    # Define a rule so we can track line numbers
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)

    t_ignore = " \t"

    # Error handling rule
    # noinspection PyMissingOrEmptyDocstring
    def t_error(self, t):
        logger.warning("Illegal character '%s'", t)
        t.lexer.skip(1)

    # noinspection PyMissingOrEmptyDocstring
    def input(self, data):
        return self.lexer.input(data)

    # noinspection PyMissingOrEmptyDocstring
    def token(self):
        return self.lexer.token()

    # def test(self, data):
    #     # Give the lexer some input
    #     self.lexer.input(data)
    #     # Tokenize
    #     while True:
    #         tok = self.lexer.token()
    #         if not tok:
    #             break  # No more input
    #         logger.info(tok)


# noinspection PyMethodMayBeStatic
class NeuralLogParser:
    """
    Class to parse NeuralLog Programs.
    """

    def __init__(self, lexer, **kwargs):
        """
        Creates a NeuralLog parser.

        :param lexer: the lexer
        :type lexer: NeuralLogLexer
        :param kwargs: optional arguments to be passed to the ply parser
        library.
        :type kwargs: dict
        """
        self.lexer = lexer
        self.tokens = lexer.tokens
        self.parser = yacc.yacc(module=self, **kwargs)

        self.for_context = deque()

        self.clauses = deque([deque()])  # type: deque[deque]

        self.predicates = set()
        self.constants = set()

        self.filename = None

    def parse(self, filename):
        """
        Parses the data.

        :param filename: the path of the file to be parsed to be parsed
        :type filename: str
        """
        self.filename = filename
        with open(filename) as data:
            self.parser.parse(input=data.read(), lexer=self.lexer)
        self.expand_placeholders()

    # noinspection DuplicatedCode
    def expand_placeholders(self):
        """
        Expands the placeholders from the Horn clauses.

        :raise BadClauseException: if a clause is malformed
        """
        expanded_clauses = deque()
        predicates_names = set()
        constants_names = set()
        predicates_names.update(map(lambda x: x.get_name(), self.predicates))
        constants_names.update(map(lambda x: x.get_name(), self.constants))

        for clause in self.clauses[0]:
            if not clause.is_template():
                expanded_clauses.append(clause)
                continue
            if not isinstance(clause, HornClause):
                raise BadClauseException(clause)
            place_holders = dict()
            for literal in clause.body:
                if literal.predicate.is_template():
                    ground_placeholders(literal.predicate.parts, place_holders,
                                        *predicates_names)
                for term in literal.terms:
                    self.process_term(term, place_holders, predicates_names,
                                      constants_names)
            solved = sorted(solve_place_holders(clause, place_holders),
                            key=lambda x: x.__str__())
            for new_clause in solved:
                if self.is_valid(new_clause):
                    expanded_clauses.append(new_clause)
                    self.predicates.add(new_clause.head.predicate)
                    predicates_names.add(new_clause.head.predicate.get_name())
                    self.add_constant_names_to_set(clause, constants_names)
                    self.add_constants(new_clause)
        self.clauses.clear()
        self.clauses.append(expanded_clauses)

    def process_term(self, term, place_holders, predicates_names,
                     constants_names):
        """
        Process the placeholder term.

        :param term: the term
        :type term: Term
        :param place_holders: the place_holders
        :type place_holders: Dict[str, Set[str]]
        :param predicates_names: the names of the predicates
        :type predicates_names: set[str]
        :param constants_names: the names of the constants
        :type constants_names: set[str]
        """
        if not term.is_template():
            return
        if isinstance(term, ListTerms):
            for sub_term in term.items:
                self.process_term(sub_term, place_holders, predicates_names,
                                  constants_names)
        else:
            if isinstance(term, Quote):
                parts = PLACE_HOLDER.split(term.get_name())
            else:
                parts = term.parts
            ground_placeholders(parts, place_holders,
                                *predicates_names, *constants_names)

    def add_constant_names_to_set(self, clause, constants_set):
        """
        Adds the constant from the clause to `constants_set`
        :param clause: the clause
        :type clause: HornClause
        :param constants_set: the set of constants
        :type constants_set: set[Term]
        """
        for term in clause.head:
            if isinstance(term, Constant) or \
                    (isinstance(term,
                                Quote) and term.is_constant()):
                constants_set.add(term.get_name())
        for literal in clause.body:
            for term in literal.terms:
                if isinstance(term, Constant) or \
                        (isinstance(term,
                                    Quote) and term.is_constant()):
                    constants_set.add(term.get_name())

    def is_valid(self, clause):
        """
        Checks if the solved clause is valid.

        It is valid if all predicates in its body already exist in the language.

        :param clause: the clause
        :type clause: HornClause
        :return: true if it is valid, false otherwise.
        :rtype: bool
        """
        if clause.is_template():
            return False
        for literal in clause.body:
            if literal.predicate not in self.predicates:
                return False

        return True

    def add_constant(self, term):
        """
        Add the term to the set of constants, if the term is a constant
        :param term: the term to be added
        :type term: Term
        """
        if isinstance(term, Constant) or \
                (isinstance(term, Quote) and term.is_constant()):
            self.constants.add(term)

    def add_constants(self, clause):
        """
        Adds the constants of the clause in the constant set.
        :param clause: the clause
        :type clause: HornClause
        """
        for term in clause.head:
            self.add_constant(term)
        for literal in clause.body:
            for term in literal.terms:
                self.add_constant(term)

    def build_atom(self, predicate, terms, start_line):
        """
        Builds the atom from the read predicate and terms.

        :param predicate: the predicates
        :type predicate: list[str]
        :param terms:
        :type terms: list[list[str]]
        :param start_line: the start line of the atom
        :type start_line: int
        :return: the atom
        :rtype: Atom
        """
        atom_terms = []
        for term in terms:
            atom_terms.append(self.build_term_from_parts(term))

        arity = len(atom_terms)
        predicate = self.build_predicate_from_parts(predicate, arity)

        provenance = FileDefinedClause(start_line, self.filename)
        return Atom(predicate, *atom_terms, provenance=provenance)

    def build_term_from_parts(self, parts):
        """
        Builds a term from the parts.

        :param parts: the parts
        :type parts: list[str]
        :return: the term
        :rtype: Term or TemplateTerm
        """
        if isinstance(parts, list):
            if len(parts) == 1 and not self.is_template(parts[0]):
                return self.extract_term(parts[0])
            else:
                return TemplateTerm(parts)
        elif isinstance(parts, ListContainer):
            if parts.container_type == ContainerTypes.LIST_OF_ARGUMENTS:
                terms = []
                for term in parts.items:
                    terms.append(self.build_term_from_parts(term))
                return ListTerms(terms)
            elif parts.container_type == ContainerTypes.PLACE_HOLDER_CONTAINER:
                if len(parts.items) == 1 and not self.is_template(parts[0]):
                    return self.extract_term(parts[0])
                return TemplateTerm(parts.items)
            else:
                raise BadArgumentException(parts)
        else:
            return self.extract_term(parts)

    def extract_term(self, term):
        """
        Extracts the term from the raw value.

        :param term: the raw term
        :type term: str, int, float
        :return: the extracted term
        :rtype: Term
        """
        if isinstance(term, float) or isinstance(term, int):
            return Number(term)
        elif term[0].isupper():
            return Variable(term)
        elif term[0].islower():
            constant = Constant(term)
            self.constants.add(constant)
            return constant
        elif self.is_quoted(term):
            quote = Quote(term)
            if quote.is_constant():
                self.constants.add(quote)
            return quote
        else:
            try:
                value = int(term)
                return Number(value)
            except ValueError:
                try:
                    value = float(term)
                    return Number(value)
                except ValueError:
                    raise BadArgumentException(term)

    def build_predicate_from_parts(self, parts, arity):
        """
        Builds a predicate from the parts.

        :param parts: the parts
        :type parts: list[str]
        :param arity: the arity
        :type arity: int
        :return: the predicate
        :rtype: Predicate or TemplatePredicate
        """
        if len(parts) == 1 and not self.is_template(parts[0]):
            parts = Predicate(parts[0], arity)
            self.predicates.add(parts)
        else:
            parts = TemplatePredicate(parts, arity)
        return parts

    def is_quoted(self, term):
        """
        Checks if `term` is a quoted string.

        :param term: the term
        :type term: str
        :return: `True` if the term is a quoted string; `False` otherwise.
        :rtype: bool
        """
        return term[0] == term[-1] and (term[0] == "\"" or term[0] == "'")

    def is_template(self, term):
        """
        Checks if `term` is a template.

        :param term: the term
        :type term: str
        :return: `True` if the term is a template; `False` otherwise.
        :rtype: bool
        """
        return '{' in term

    def solve_for_context(self):
        """
        Solves the clauses, from the current for, by
        replacing their variable by the values defined in the for statement.

        :return: the for context
        :rtype: tuple[str, list]
        """
        context = self.for_context.pop()
        clauses = self.clauses.pop()
        key = context[0]
        values = context[1]
        for value in values:
            key_dict = KeyDict({key: value})
            for clause in clauses:
                self.clauses[-1].append(
                    self.solve_clause_for_context(clause, key_dict))

        return context

    def solve_clause_for_context(self, clause, key_dict):
        """
        Solves the clause by replacing the variable by the values defined in
        the for statement.

        :param clause: the clause
        :type clause: AtomClause or HornClause
        :param key_dict: the dictionary with the values for the for variables
        :type key_dict: KeyDict
        :return: the solved clause
        :rtype: AtomClause or HornClause
        """
        if not clause.is_template():
            return clause
        if isinstance(clause, AtomClause):
            return AtomClause(
                self.solve_atom_for_context(clause.atom, key_dict))
        if isinstance(clause, HornClause):
            head = self.solve_atom_for_context(clause.head, key_dict)
            body = []
            for literal in clause.body:
                atom = self.solve_atom_for_context(literal, key_dict)
                body.append(Literal(atom, negated=literal.negated))
            return HornClause(head, *body)

    def solve_atom_for_context(self, atom, key_dict):
        """
        Solves the atom by replacing the variable by the values defined in
        the for statement.

        :param atom: the atom
        :type atom: atom
        :param key_dict: the dictionary with the values for the for variables
        :type key_dict: KeyDict
        :return: the solved atom
        :rtype: Atom
        """
        predicate = atom.predicate
        if isinstance(predicate, TemplatePredicate):
            parts = self.solve_placeholders(predicate.parts, key_dict)
            predicate = self.build_predicate_from_parts(parts,
                                                        predicate.arity)
        solved_terms = []
        for term in atom.terms:
            if term.is_template():
                if isinstance(term, Quote):
                    term = Quote(term.quote +
                                 term.value.format_map(key_dict)
                                 + term.quote)
                else:
                    parts = self.solve_placeholders(term.parts, key_dict)
                    term = self.build_term_from_parts(parts)
            solved_terms.append(term)
        return Atom(predicate, *solved_terms, weight=atom.weight,
                    provenance=atom.provenance)

    def solve_placeholders(self, parts, key_dict):
        """
        Solves the placeholders from the for.

        :param parts: the parts of the term to be solved.
        :type parts: list[str]
        :param key_dict: the dictionary with the values for the for variables
        :type key_dict: KeyDict
        :return: the terms
        :rtype: list[str]
        """
        solved_parts = []
        joint_part = ""
        for part in parts:
            if self.is_template(part):
                key = part[1:-1]
                if key in key_dict:
                    joint_part += str(key_dict[key])
                else:
                    if joint_part != "":
                        solved_parts.append(str(joint_part))
                        joint_part = ""
                    solved_parts.append(part)
            else:
                joint_part += part
        if joint_part != "":
            solved_parts.append(str(joint_part))

        return solved_parts

    def get_clauses(self):
        """
        Gets the clauses parsed by the parser.

        :return: the parsed clauses
        :rtype: collections.Collection[Clause]
        """
        return self.clauses[0]

    def p_program(self, p):
        """program : statement
                   | program statement"""
        pass

    def p_statement(self, p):
        """statement : for_loop
                     | clause"""
        pass

    def p_for_loop(self, p):
        """for_loop : for_loop_init program for_loop_end"""
        pass

    def p_for_loop_init(self, p):
        """for_loop_init : FOR_LOOP for_variable IN_TOKEN for_header DO_TOKEN"""
        p[0] = (p[2], p[4])
        self.for_context.append(p[0])
        self.clauses.append(deque())
        logger.log(5, "for_loop_init:\t%s", p[0])

    def p_for_loop_end(self, p):
        """for_loop_end : DONE_TOKEN"""
        p[0] = self.solve_for_context()
        logger.log(5, "for_loop_end:\t%s", p[0])

    def p_for_header_terms(self, p):
        """for_header : for_terms"""
        p[0] = p[1]

    def p_for_header_range(self, p):
        """for_header : for_range"""
        p[0] = list(range(p[1][0], p[1][1] + 1))

    def p_fact_clause(self, p):
        """clause : atom END_OF_CLAUSE
                  | weighted_atom END_OF_CLAUSE"""
        clause = AtomClause(p[1])
        p[0] = clause
        self.clauses[-1].append(clause)
        logger.log(5, "clause:\t%s", p[0])

    def p_clause(self, p):
        """clause : horn_clause END_OF_CLAUSE"""
        clause = p[1]
        p[0] = clause
        self.clauses[-1].append(clause)
        logger.log(5, "clause:\t%s", p[0])

    def p_propositional_atom(self, p):
        """atom : predicate"""
        p.set_lineno(0, p.lineno(1))
        p[0] = self.build_atom(p[1], [], p.lineno(1))

    def p_atom(self, p):
        """atom : predicate list_of_arguments"""
        p.set_lineno(0, p.lineno(1))
        p[0] = self.build_atom(p[1], p[2], p.lineno(1))

    def p_weighted_atom(self, p):
        """weighted_atom : number WEIGHT_SEPARATOR atom"""
        p[0] = p[3]  # type: Atom
        p[0].weight = p[1]

    def p_horn_clause(self, p):
        """horn_clause : atom IMPLICATION_SIGN body"""
        provenance = FileDefinedClause(p.lineno(1), self.filename)
        p[0] = HornClause(p[1], *p[3], provenance=provenance)

    def p_e_horn_clause(self, p):
        """horn_clause : atom IMPLICATION_SIGN"""
        provenance = FileDefinedClause(p.lineno(1), self.filename)
        p[0] = HornClause(p[1], provenance=provenance)

    def p_predicate(self, p):
        """predicate : TERM
                     | PLACE_HOLDER"""
        p[0] = [p[1]]
        p.set_lineno(0, p.lineno(1))
        logger.log(5, "predicate:\t%s", p[0])

    def p_predicate_int(self, p):
        """predicate : INTEGER"""
        p[0] = [str(p[1])]
        p.set_lineno(0, p.lineno(1))
        logger.log(5, "predicate:\t%s", p[0])

    def p_r_predicate(self, p):
        """predicate : predicate TERM
                     | predicate PLACE_HOLDER"""
        p[0] = p[1] + [p[2]]
        p.set_lineno(0, p.lineno(1))
        logger.log(5, "predicate:\t%s", p[0])

    def p_r_predicate_int(self, p):
        """predicate : predicate INTEGER"""
        p[0] = p[1] + [str(p[2])]
        p.set_lineno(0, p.lineno(1))
        logger.log(5, "predicate:\t%s", p[0])

    def p_body(self, p):
        """body : literal"""
        p[0] = [p[1]]
        logger.log(5, "body:\t%s", p[0])

    def p_r_body(self, p):
        """body : body ITEM_SEPARATOR literal"""
        p[0] = p[1] + [p[3]]
        logger.log(5, "body:\t%s", p[0])

    def p_literal(self, p):
        """literal : atom"""
        p[0] = Literal(p[1], negated=False)
        logger.log(5, "literal:\t%s", p[0])

    def p_n_literal(self, p):
        """literal : NEGATION atom"""
        p[0] = Literal(p[2], negated=True)
        logger.log(5, "literal:\t%s", p[0])

    def p_list_of_arguments(self, p):
        """list_of_arguments : OPEN_ARGUMENTS arguments CLOSE_ARGUMENTS"""
        p[0] = p[2]
        logger.log(5, "list_of_arguments:\t%s", p[0])

    def p_arguments(self, p):
        """arguments : argument """
        p[0] = [p[1]]

    def p_r_arguments(self, p):
        """arguments : arguments ITEM_SEPARATOR argument"""
        p[0] = p[1] + [p[3]]

    def p_argument(self, p):
        """argument : number
                    | term"""
        p[0] = p[1]

    def p_list_argument(self, p):
        """argument : OPEN_LIST_ARGUMENT arguments CLOSE_LIST_ARGUMENT"""
        p[0] = ListContainer(ContainerTypes.LIST_OF_ARGUMENTS, p[2])

    def p_term_quoted(self, p):
        """term : QUOTED"""
        p[0] = p[1]

    def p_term_place_holder(self, p):
        """term : place_holder_term"""
        p[0] = ListContainer(ContainerTypes.PLACE_HOLDER_CONTAINER, p[1])

    def p_place_holder_term(self, p):
        """place_holder_term : PLACE_HOLDER
                             | TERM"""
        p[0] = [p[1]]

    def p_place_holder_term_int(self, p):
        """place_holder_term : INTEGER"""
        p[0] = [str(p[1])]

    def p_r_place_holder_term(self, p):
        """place_holder_term : place_holder_term PLACE_HOLDER
                             | place_holder_term TERM"""
        p[0] = p[1] + [p[2]]

    def p_r_place_holder_term_int(self, p):
        """place_holder_term : place_holder_term INTEGER"""
        p[0] = p[1] + [str(p[2])]

    def p_for_variable(self, p):
        """for_variable : TERM"""
        p[0] = p[1]

    def p_for_terms(self, p):
        """for_terms : for_term"""
        p[0] = [p[1]]

    def p_r_for_terms(self, p):
        """for_terms : for_terms for_term"""
        p[0] = p[1] + [p[2]]

    def p_for_term(self, p):
        """for_term : TERM
                    | QUOTED
                    | number"""
        p[0] = p[1]

    def p_for_range(self, p):
        """for_range : OPEN_CURLY_BRACES for_range_indices CLOSE_CURLY_BRACES"""
        p[0] = p[2]

    def p_for_range_indices(self, p):
        """for_range_indices : INTEGER RANGER_SEPARATOR INTEGER"""
        p[0] = (p[1], p[3])

    def p_integer_number(self, p):
        """number : INTEGER"""
        p[0] = int(p[1])

    def p_float_number(self, p):
        """number : SCIENTIFIC_NUMBER
                  | DECIMAL"""
        p[0] = float(p[1])

    # Error rule for syntax errors
    # noinspection PyMissingOrEmptyDocstring
    def p_error(self, p):
        logger.warning("Syntax error in input:\t%s", p)
