"""
Define the classes to represent the knowledge of the system.
"""

import re

TRAINABLE_KEY = "$"

WEIGHT_SEPARATOR = "::"
NEGATION_KEY = "not"
IMPLICATION_SIGN = ":-"
END_SIGN = "."
PLACE_HOLDER = re.compile("({[a-zA-Z0-9_-]+})")


def get_term_from_string(string):
    """
    Transforms the string into a term. A variable if it starts with an upper
    case letter, or a constant, otherwise.

    :param string: the string
    :type string: str
    :return: the term
    :rtype: Term
    """
    if string[0].isupper():
        return Variable(string)
    elif string[0].islower():
        return Constant(string)
    elif string[0] == string[-1] and (string[0] == "'" or string[0] == '"'):
        return Quote(string)
    else:
        raise TermMalformedException


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
            terms.append(get_term_from_string(argument))
        elif isinstance(argument, float) or isinstance(argument, int):
            terms.append(Number(argument))
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


class ClauseMalformedException(Exception):
    """
    Represents an term malformed exception.
    """

    def __init__(self) -> None:
        """
        Creates an term malformed exception.
        """
        super().__init__("Clause malformed, the clause must be an atom, "
                         "a weighted atom or a Horn clause.")


class BadArgumentException(Exception):
    """
    Represents an bad argument exception.
    """

    def __init__(self, value) -> None:
        """
        Creates an term malformed exception.
        """
        super().__init__("Expected a number or a term, got {}".format(value))


class Term:
    """
    Represents a logic term.
    """

    def __init__(self, value):
        """
        Creates a logic term.

        :param value: the value of the term
        :type value: str or float
        """
        self.value = value

    def get_name(self):
        """
        Returns the name of the term.
        :return: the name of the term
        :rtype: str
        """
        return self.value

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

    def key(self):
        """
        Specifies the keys to be used in the equals and hash functions.
        :return: the keys
        :rtype: Any
        """
        return self.value

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        if isinstance(other, Term):
            return self.key() == other.key()
        return False

    def __str__(self):
        return self.value

    __repr__ = __str__


class Constant(Term):
    """
    Represents a logic constant.
    """

    def __init__(self, value):
        """
        Creates a logic constant.

        :param value: the name of the constant
        :type value: str
        """
        super().__init__(value)

    # noinspection PyMissingOrEmptyDocstring
    def is_constant(self):
        return True

    def __str__(self):
        return self.value

    __repr__ = __str__


class Variable(Term):
    """
    Represents a logic variable.
    """

    def __init__(self, value):
        """
        Creates a logic variable.

        :param value: the name of the variable
        :type value: str
        """
        super().__init__(value)

    # noinspection PyMissingOrEmptyDocstring
    def is_constant(self):
        return False


class Quote(Term):
    """
    Represents a quoted term, that might contain a template.
    """

    def __init__(self, value):
        """
        Creates a quoted term.

        :param value: the value of the term.
        :type value: str
        """
        super().__init__(value[1:-1])
        self._is_template = PLACE_HOLDER.search(value) is not None
        self.quote = value[0]

    # noinspection PyMissingOrEmptyDocstring
    def is_constant(self):
        return not self._is_template

    # noinspection PyMissingOrEmptyDocstring
    def is_template(self):
        return self._is_template

    def __str__(self):
        return self.quote + super().__str__() + self.quote


class Number(Term):
    """
    Represents a number term.
    """

    def __init__(self, value):
        super().__init__(value)

    # noinspection PyMissingOrEmptyDocstring
    def get_name(self):
        return str(self.value)

    # noinspection PyMissingOrEmptyDocstring
    def is_constant(self):
        return True

    # noinspection PyMissingOrEmptyDocstring
    def is_template(self):
        return False

    def __str__(self):
        return str(self.value)


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

    def key(self):
        """
        Specifies the keys to be used in the equals and hash functions.
        :return: the keys
        :rtype: Any
        """
        return self.name, self.arity

    def get_name(self):
        """
        Returns the name of the predicate.

        :return: the name of the predicate
        :rtype: str
        """
        return self.name

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.key() == other.key()
        return False

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

    def is_grounded(self):
        """
        Returns true if the clause is grounded. A clause is grounded if all
        its terms are constants.
        :return: true if the clause is grounded, false otherwise
        :rtype: bool
        """
        pass

    def key(self):
        """
        Specifies the keys to be used in the equals and hash functions.
        :return: the keys
        :rtype: Any
        """
        pass

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.key() == other.key()
        return False


class Atom(Clause):
    """
    Represents a logic atom.
    """

    def __init__(self, predicate, *args, weight=1.0) -> None:
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

    # noinspection PyMissingOrEmptyDocstring
    def is_grounded(self):
        for term in self.terms:
            if isinstance(term, Term) and not term.is_constant():
                return False
        return True

    def arity(self):
        """
        Returns the arity of the atom.

        :return: the arity of the atom
        :rtype: int
        """
        return self.predicate.arity

    # noinspection PyMissingOrEmptyDocstring
    def key(self):
        return self.weight, self.predicate, tuple(self.terms)

    def __str__(self):
        if self.terms is None or len(self.terms) == 0:
            if self.weight == 1.0:
                return self.predicate.name
            else:
                return "{}::{}".format(self.weight, self.predicate.name)

        atom = "{}({})".format(self.predicate.name,
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

    def __init__(self, atom, negated=False, trainable=False) -> None:
        """
        Creates a logic literal from an atom.

        :param atom: the atom
        :type atom: Atom
        :param negated: if the literal is negated
        :type negated: bool
        :param trainable: if the literal is to be trained
        :type trainable: bool
        :raise AtomMalformedException in case the number of terms differs
        from the arity of the predicate
        """
        super().__init__(atom.predicate, *atom.terms, weight=atom.weight)
        self.negated = negated
        self.trainable = trainable

    # noinspection PyMissingOrEmptyDocstring
    def key(self):
        # noinspection PyTypeChecker
        return (self.negated, self.trainable) + super().key()

    def __str__(self):
        atom = super().__str__()
        if self.trainable:
            atom = TRAINABLE_KEY + atom
        if self.negated:
            return "{} {}".format(NEGATION_KEY, atom)
        return atom


class AtomClause(Clause):
    """
    Represents an atom clause, an atom that is also a clause. As such,
    it is written with a `END_SIGN` at the end.
    """

    def __init__(self, atom) -> None:
        """
        Creates an atom clause
        :param atom: the atom
        :type atom: Atom
        """
        super().__init__()
        self.atom = atom

    # noinspection PyMissingOrEmptyDocstring
    def is_template(self):
        return self.atom.is_template()

    # noinspection PyMissingOrEmptyDocstring
    def is_grounded(self):
        return self.atom.is_grounded()

    # noinspection PyMissingOrEmptyDocstring
    def key(self):
        return self.atom.key()

    def __str__(self) -> str:
        return self.atom.__str__() + END_SIGN


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

    # noinspection PyMissingOrEmptyDocstring
    def key(self):
        key_tuple = self.head.key()
        if self.body is not None and len(self.body) > 0:
            for literal in self.body:
                key_tuple += literal.key()
        return tuple(key_tuple)

    def __str__(self):
        body = Literal(Atom("true")) if self.body is None else self.body

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
