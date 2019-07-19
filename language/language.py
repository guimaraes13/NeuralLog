"""
Define the classes to represent the knowledge of the system.
"""
WEIGHT_SEPARATOR = "::"
NEGATION_KEY = "not"
IMPLICATION_SIGN = ":-"
END_SIGN = "."


def build_terms(arguments):
    """
    Builds a list of terms from the arguments.

    :param arguments: the arguments
    :type arguments: list[Term or str]
    :raise TermMalformedException if the argument is neither a term nor a str
    :return: a list of terms
    :rtype: list[Term]
    """
    terms = []
    for argument in arguments:
        if isinstance(argument, Term):
            terms.append(argument)
        elif isinstance(argument, str):
            if argument[0].isupper():
                terms.append(Variable(argument))
            else:
                terms.append(Constant(argument))
        else:
            raise TermMalformedException()

    return terms


class AtomMalformedException(Exception):
    """
    Represents an atom malformed exception.
    """

    def __init__(self, expected, found) -> None:
        """
        Creates an atom malformed exception.

        :param expected: the number of arguments expected
        :type expected: int
        :param found: the number of arguments found
        :type found: int
        """
        super().__init__("Atom malformed, the predicate expects "
                         "{} argument(s) but {} found".format(expected, found))


class TermMalformedException(Exception):
    """
    Represents an term malformed exception.
    """

    def __init__(self) -> None:
        """
        Creates an term malformed exception.
        """
        super().__init__("Term malformed, the argument must be either a term "
                         "or a string.")


class Term:
    """
    Represents a logic term.
    """

    def __init__(self, name):
        """
        Creates a logic term.

        :param name: the name of the term
        :type name: str
        """
        self.name = name

    def is_constant(self):
        """
        Returns true if the term is a constant.

        :return: true if it is a constant, false otherwise
        :rtype: bool
        """
        return False

    def is_template(self):
        """
        Returns true if the term is a template to be instantiated based on
        the knowledge base.
        :return: True if the term is a template, false otherwise
        :rtype: bool
        """
        return False

    def __str__(self):
        return self.name

    __repr__ = __str__


class Constant(Term):
    """
    Represents a logic constant.
    """

    def __init__(self, name):
        """
        Creates a logic constant.

        :param name: the name of the constant
        :type name: str
        """
        super().__init__(name)

    # noinspection PyMissingOrEmptyDocstring
    def is_constant(self):
        return True

    def __str__(self):
        return self.name

    __repr__ = __str__


class Variable(Term):
    """
    Represents a logic variable.
    """

    def __init__(self, name):
        """
        Creates a logic variable.

        :param name: the name of the variable
        :type name: str
        """
        super().__init__(name)

    # noinspection PyMissingOrEmptyDocstring
    def is_constant(self):
        return False


class TemplateTerm(Term):
    """
    Defines a template term, a term with variable parts to be instantiated
    based on the knowledge base.
    """

    def __init__(self, parts):
        """
        Creates a template term.

        :param parts: the parts of the term.
        :type parts: list[str]
        """
        super().__init__("".join(parts))
        self.parts = parts

    # noinspection PyMissingOrEmptyDocstring
    def is_constant(self):
        return False

    # noinspection PyMissingOrEmptyDocstring
    def is_template(self):
        return True


class Predicate:
    """
    Represents a logic predicate.
    """

    def __init__(self, name, arity=0):
        """
        Creates a logic predicate.

        :param name: the name of the predicate
        :type name: str
        :param arity: the arity of the predicate
        :type arity: int
        """
        self.name = name
        self.arity = arity

    def is_template(self):
        """
        Returns true if the predicate is a template to be instantiated based on
        the knowledge base.
        :return: True if the predicate is a template, false otherwise
        :rtype: bool
        """
        return False

    def __str__(self):
        return "{}/{}".format(self.name, self.arity)

    __repr__ = __str__


class TemplatePredicate(Predicate):
    """
    Defines a template predicate, a predicate with variable parts to be
    instantiated based on the knowledge base.
    """

    def __init__(self, parts, arity=0):
        """
        Creates a template predicate.

        :param parts: the parts of the predicate.
        :type parts: list[str]
        """
        super().__init__("".join(parts), arity=arity)
        self.parts = parts

    # noinspection PyMissingOrEmptyDocstring
    def is_template(self):
        return True


class Clause:
    """
    Represents a logic clause.
    """

    def is_template(self):
        """
        Returns true if the clause is a template to be instantiated based on
        the knowledge base.
        :return: True if the clause is a template, false otherwise
        :rtype: bool
        """
        return False


class Atom(Clause):
    """
    Represents a logic atom.
    """

    def __init__(self, predicate, weight=1.0, *args) -> None:
        """
        Creates a logic atom.

        :param predicate: the predicate
        :type predicate: str or Predicate
        :param args: the list of terms, if any
        :type args: Term or str
        :param weight: the weight of the atom
        :type weight: float
        :raise AtomMalformedException in case the number of terms differs
        from the arity of the predicate
        """
        if isinstance(predicate, Predicate):
            self.predicate = predicate
            if predicate.arity != len(args):
                raise AtomMalformedException(predicate.arity,
                                             len(args))
        else:
            self.predicate = Predicate(predicate, len(args))

        self.weight = weight
        # noinspection PyTypeChecker
        self.terms = build_terms(args)

    def __getitem__(self, item):
        return self.terms[item]

    def arity(self):
        """
        Returns the arity of the atom.

        :return: the arity of the atom
        :rtype: int
        """
        return self.predicate.arity

    def __str__(self):
        if self.terms is None or len(self.terms) == 0:
            return self.predicate.name

        atom = "{}({})".format(self.predicate,
                               ", ".join(map(lambda x: str(x), self.terms)))
        if self.weight != 1.0:
            return "{}{}{}".format(self.weight, WEIGHT_SEPARATOR, atom)
        return atom

    # noinspection PyMissingOrEmptyDocstring
    def is_template(self):
        if self.predicate.is_template():
            return True
        if self.terms is None or len(self.terms) == 0:
            return False
        for term in self.terms:
            if term.is_template():
                return True

        return False

    __repr__ = __str__


class Literal(Atom):
    """
    Represents a logic literal.
    """

    def __init__(self, predicate, weight=1.0,
                 negated=False, learnable=False, *args) -> None:
        """
        Creates a logic literal.

        :param predicate: the predicate
        :type predicate: str or Predicate
        :param weight: the weight of the atom
        :type weight: float
        :param negated: if the literal is negated
        :type negated: bool
        :param learnable: if the literal is to be learned
        :type learnable: bool
        :param args: the list of terms, if any
        :type args: Term or str
        :raise AtomMalformedException in case the number of terms differs
        from the arity of the predicate
        """
        super().__init__(predicate, weight, *args)
        self.negated = negated
        self.learnable = learnable

    def __str__(self):
        atom = super().__str__()
        if self.negated:
            return "{} {}".format(NEGATION_KEY, atom)
        return atom


class HornClause(Clause):
    """
    Represents a logic horn clause.
    """

    def __init__(self, head, *body) -> None:
        """
        Creates a Horn clause.

        :param head: the head
        :type head: Atom
        :param body: the list of literal, if any
        :type body: Literal
        """
        self.head = head
        self.body = list(body)

    def __getitem__(self, item):
        return self.body[item]

    def __str__(self):
        body = Literal("true") if self.body is None else self.body

        return "{} {} {}{}".format(self.head, IMPLICATION_SIGN,
                                   ", ".join(map(lambda x: str(x), body)),
                                   END_SIGN)

    # noinspection PyMissingOrEmptyDocstring
    def is_template(self):
        if self.head.is_template():
            return True
        if self.body is None or len(self.body) == 0:
            return False
        for literal in self.body:
            if literal.is_template():
                return True

        return False

    __repr__ = __str__


class ForLoop:
    """
    Represents a for loop.
    """

    def __init__(self, variable, for_terms, content):
        """
        Creates the representation of a for loop.

        :param variable: the variable of the for
        :type variable: str
        :param for_terms: the items for the variable to iterate over
        :type for_terms: list[str]
        :param content: the content of the for
        :type content: list[ForLoop or Clause]
        """
        self.variable = variable
        self.for_terms = for_terms
        self.content = content
