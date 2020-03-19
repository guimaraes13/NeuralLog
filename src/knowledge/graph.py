"""
Represents a graph from a clause.
"""
from collections import deque
from typing import Dict, List, Set

from src.knowledge.program import ANY_PREDICATE_NAME
from src.language.language import Term, Literal, Atom


class RulePathFinder:
    """
    Represents a rule graph
    """
    edge_literals: Dict[Term, List[Literal]]
    loop_literals: Dict[Term, List[Literal]]

    def __init__(self, clause):
        """
        Creates a rule graph.

        :param clause: the clause
        :type clause: HornClause
        """
        self.clause = clause
        self._build_graph()

    def _build_graph(self):
        """
        Builds the graph from the clause.

        :return: the graph
        :rtype: dict[Term, list[Literal]]
        """
        self.edge_literals = dict()
        self.loop_literals = dict()
        for literal in self.clause.body:
            arity = literal.arity()
            if arity == 0:
                continue
            elif arity == 1 or literal.terms[0] == literal.terms[-1]:
                self.loop_literals.setdefault(
                    literal.terms[0], []).append(literal)
            else:
                for i in range(arity):
                    self.edge_literals.setdefault(
                        literal.terms[i], []).append(literal)

    # TODO: populate the graph with the corresponding edges
    def find_clause_paths(self, output_index):
        """
        Finds the paths in the clause.

        :param output_index: the index of the output term
        :type output_index: int
        :return: the completed paths between the terms of the clause and the
        remaining grounded literals
        :rtype: (List[RulePath], List[Literal])
        """
        # Defining variables
        sources = list(self.clause.head.terms)
        destination = sources.pop(output_index)

        paths = []
        sources_set = set(sources)
        all_visited = set()
        for source in sources:
            visited = set()
            paths += self.find_forward_paths(
                source, destination, visited, sources_set)
            all_visited.update(visited)

        if not all_visited.issuperset(self.clause.body):
            # Finding backward paths
            source = destination
            for destination in sources:
                visited = set(all_visited)
                # visited = set()
                backward_paths = self.find_forward_paths(
                    source, destination, visited)
                path_set = set(paths)
                for backward_path in backward_paths:
                    reversed_path = backward_path.reverse()
                    if reversed_path not in path_set:
                        path_set.add(reversed_path)
                        paths.append(reversed_path)
                all_visited.update(visited)

        ground_literals = self.get_disconnected_literals(all_visited)

        return paths, ground_literals

    def get_disconnected_literals(self, connected_literals):
        """
        Gets the literals from the `clause` which are disconnected from the
        source and destination variables but are grounded.

        :param connected_literals: the set of connected literals
        :type connected_literals: Set[Literal]
        :return: the list of disconnected literals
        :rtype: List[Literal]
        """
        ground_literals = []
        for literal in self.clause.body:
            if literal in connected_literals or not literal.is_grounded():
                continue
            ground_literals.append(literal)
        return ground_literals

    # noinspection DuplicatedCode
    def find_forward_paths(self, source, destination, visited_literals,
                           all_sources=None):
        """
        Finds all forward paths from `source` to `destination` by using the
        literals in the clause.
        If the destination cannot be reached, it includes a special `any`
        predicated to connect the path.

        :param source: the source of the paths
        :type source: Term
        :param destination: the destination of the paths
        :type destination: Term
        :param visited_literals: the set of visited literals
        :type visited_literals: Set[Literal]
        :param all_sources: the set of all sources
        :type all_sources: Set[Term]
        :return: the completed forward paths between source and destination
        :rtype: List[RulePath]
        """
        if all_sources is None:
            all_sources = {source}
        partial_paths = deque()  # type: deque[RulePath]

        initial_path = self.build_initial_path(source, visited_literals)
        for literal in self.edge_literals.get(source, []):
            if literal in visited_literals:
                continue
            new_path = initial_path.new_path_with_item(literal)
            if new_path is not None:
                visited_literals.add(literal)
                if new_path.path_end() != destination:
                    for loop in self.loop_literals.get(new_path.path_end(), []):
                        new_path.append(loop)
                        visited_literals.add(loop)
                partial_paths.append(new_path)
        if len(partial_paths) == 0:
            partial_paths.append(initial_path)

        return self.find_paths(
            partial_paths, destination, visited_literals, all_sources)

    @staticmethod
    def complete_path_with_any(dead_end_paths, destination):
        """
        Completes the path by appending the special `any` predicate between the
        end of the path and the destination.

        :param dead_end_paths: the paths to be completed
        :type dead_end_paths: collections.Iterable[RulePath]
        :param destination: the destination
        :type destination: Term
        :return: the completed paths
        :rtype: List[RulePath]
        """
        completed_paths = []
        for path in dead_end_paths:
            any_literal = Literal(Atom(ANY_PREDICATE_NAME,
                                       path.path_end(), destination))
            path.append(any_literal)
            completed_paths.append(path)
        return completed_paths

    def build_initial_path(self, source, visited_literals):
        """
        Builds a path with its initial literals, if any.

        :param source: the source of the path
        :type source: Term
        :param visited_literals: the set of visited literals
        :type visited_literals: set[Literal]
        :return: the path or `None`
        :rtype: RulePath
        """
        loop_literals = self.loop_literals.get(source, [])
        path = RulePath(source, loop_literals)
        visited_literals.update(path.literals)
        return path

    # noinspection DuplicatedCode
    def find_paths(self, partial_paths, destination, visited_literals,
                   all_sources):
        """
        Finds the paths from `partial_paths` to `destination` by appending the
        literals from clause.

        :param partial_paths: The initial partial paths
        :type partial_paths: deque[RulePath]
        :param destination: the destination term
        :type destination: Term
        :param visited_literals: the visited literals
        :type visited_literals: Set[Literal]
        :param all_sources: the set of all sources
        :type all_sources: Set[Term]
        :return: the completed paths
        :rtype: List[RulePath]
        """
        completed_paths = deque()  # type: deque[RulePath]
        while len(partial_paths) > 0:
            size = len(partial_paths)
            for i in range(size):
                path = partial_paths.popleft()
                path_end = path.path_end()

                if path_end == destination:
                    if path.source != destination:
                        for loop in self.loop_literals.get(destination, []):
                            path.append(loop)
                            visited_literals.add(loop)
                    completed_paths.append(path)
                    continue

                possible_edges = self.edge_literals.get(path_end, [])
                not_added_path = True
                for literal in possible_edges:
                    new_path = path.new_path_with_item(literal)
                    if new_path is None:
                        continue
                    end = new_path.path_end()
                    if end != path.source and end in all_sources:
                        continue
                    if end != destination:
                        for loop in self.loop_literals.get(end, []):
                            new_path.append(loop)
                            visited_literals.add(loop)
                    partial_paths.append(new_path)
                    # noinspection PyTypeChecker
                    visited_literals.add(literal)
                    not_added_path = False

                if not_added_path:
                    if path.source == destination:
                        completed_paths.append(path)
                    else:
                        any_literal = Literal(Atom(
                            ANY_PREDICATE_NAME, path.path_end(), destination))
                        path.append(any_literal)
                        partial_paths.append(path)
        return completed_paths

    def __str__(self):
        return "[{}] {}".format(self.__class__.__name__, self.clause.__str__())

    def __repr__(self):
        return "[{}] {}".format(self.__class__.__name__, self.clause.__repr__())


class RulePath:
    """
    Represents a rule path.
    """

    source: Term
    "The source term"

    path: List[Literal] = list()
    "The path of literals"

    literals: Set[Literal] = set()
    "The set of literals in the path"

    terms: Set[Term]
    "The set of all terms in the path"

    input_indices: List[int]
    "The list with the indices of the input term for each literal."

    output_indices: List[int]
    "The list with the indices of the output term for each literal."

    def __init__(self, source, path=()):
        """
        Initializes a path.

        :param source: The source term
        :type source: Term
        :param path: the path
        :type path: collections.Sequence[Literal]
        """
        self.source = source
        self.path = list()
        self.literals = set()
        self.terms = set()
        self.input_indices = list()
        self.output_indices = list()

        for literal in path:
            self.append(literal)

    def path_end(self):
        """
        Gets the term at the end of the path.

        :return: the term at the end of the path
        :rtype: Term
        """
        if len(self.path) == 0:
            return self.source
        return self.path[-1].terms[self.output_indices[-1]]

    def append(self, item, force_output=None):
        """
        Appends the item to the end of the path, if it is not in the path yet.

        :param item: the item
        :type item: Literal
        :param force_output: forces the output term to be at this index
        :type force_output: int
        :return: True, if the item has been appended to the path; False,
        otherwise.
        :rtype: bool
        """
        if item is None:
            return False

        arity = item.arity()
        if arity == 1:
            input_index = 0
            out_index = 0
        elif arity == 2:
            if item.terms[-1] == self.path_end():
                # inverted literal case
                input_index = 1
                out_index = 0
                if item.predicate.name == ANY_PREDICATE_NAME:
                    # if the literal is any, reverse the terms and make it
                    # not inverted. Since any^{-1} == any
                    item = Literal(
                        Atom(item.predicate, *list(reversed(item.terms)),
                             weight=item.weight), negated=item.negated)
            else:
                input_index = 0
                out_index = 1
        else:
            # high arity item
            # for now, assumes that the output of a literal with arity bigger
            # than 2 is always the last term
            input_index = item.terms.index(self.path_end())
            out_index = force_output if force_output is not None else arity - 1

        output_variable = item.terms[out_index]
        # Detects if there is a loop in the path
        # If it is a loop, the path is invalid
        #   assumes that all terms of a literal with arity bigger than two are
        #   all different among each other
        # It also avoids paths return through a literal with arity bigger than 2

        if arity != 1 and \
                item.terms[0] != item.terms[-1] and \
                output_variable in self.terms:
            return False

        self.path.append(item)
        self.literals.add(item)
        self.terms.update(item.terms)
        self.input_indices.append(input_index)
        self.output_indices.append(out_index)
        return True

    def new_path_with_item(self, item):
        """
        Creates a new path with `item` at the end.

        :param item: the item
        :type item: Literal or None
        :return: the new path, if its is possible to append the item; None,
        otherwise
        :rtype: RulePath or None
        """
        path = RulePath(self.source, self.path)
        return path if path.append(item) else None

    def all_paths_with_item(self, item):
        """
        Creates all possible new paths with `item` at the end.

        :param item: the item
        :type item: Literal or None
        :return: the new path, if its is possible to append the item; None,
        otherwise
        :rtype: list[RulePath]
        """
        paths = []
        for i in range(len(item.terms)):
            if item.terms[i] == self.path_end():
                continue
            path = RulePath(self.source, self.path)
            if path.append(item, i):
                paths.append(path)
        return paths

    def reverse(self):
        """
        Gets a reverse path.

        :return: the reverse path
        :rtype: RulePath
        """
        source = self.path[-1].terms[self.output_indices[-1]]
        reversed_path = RulePath(source)

        reversed_path.path = list(reversed(self.path))
        reversed_path.literals = set(self.literals)
        reversed_path.terms = set(self.terms)
        reversed_path.input_indices = list(reversed(self.output_indices))
        reversed_path.output_indices = list(reversed(self.input_indices))
        return reversed_path
        # return RulePath2(source, list(reversed(self.path)))

    def __getitem__(self, item):
        return self.path.__getitem__(item)

    def __len__(self):
        return self.path.__len__()

    def __str__(self):
        message = []
        for i in range(0, len(self.path)):
            prefix = self.path[i].predicate.name
            iterator = list(map(lambda x: x.value, self.path[i].terms))
            if self.input_indices[i] > self.output_indices[i]:
                if self.path[i].predicate.name != ANY_PREDICATE_NAME:
                    prefix += "^{-1}"
                iterator = reversed(iterator)

            prefix += "("
            prefix += ", ".join(iterator)
            prefix += ")"
            message.append(prefix)

        return ", ".join(message)

    __repr__ = __str__

    def __hash__(self):
        return hash((self.source, tuple(self.path)))

    def __eq__(self, other):
        if not isinstance(other, RulePath):
            return False
        return self.source == other.source, self.path == other.path

    def get_input_term(self, index):
        """
        Returns the input term of the literal at `index`.

        :param index: the index
        :type index: int
        :return: the input term at the index
        :rtype: Term
        """

        return self.path[index].terms[self.input_indices[index]]

    def get_output_term(self, index):
        """
        Returns the output term of the literal at `index`.

        :param index: the index
        :type index: int
        :return: the output term at the index
        :rtype: Term
        """

        return self.path[index].terms[self.output_indices[index]]

    def is_loop(self, index):
        """
        Returns `True` if the literal at `index` is a loop.

        :param index: the index of the literal
        :type index: int
        :return: `True`, if the literal at `index` is a loop; otherwise, `False`
        :rtype: bool
        """
        return self.get_input_term(index) == self.get_output_term(index)


class Edge:
    """
    Represents an edge in the graph.
    """

    def __init__(self, literal, input_index, output_index):
        """
        Creates the edge.

        :param literal: the literal of the edge
        :type literal: Literal
        :param input_index: the index of the input term
        :type input_index: int
        :param output_index: the index of the output term
        :type output_index: int
        """
        self.literal = literal
        self.input_index = input_index
        self.output_index = output_index

    def __hash__(self):
        return hash((self.literal, self.input_index, self.output_index))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False

        if self.literal != other.literal:
            return False

        if self.input_index != other.input_index:
            return False

        if self.output_index != other.output_index:
            return False

        return True


class RuleGraph:
    """
    Represents a rule as a graph.
    """

    sources: List[Term] = list()
    "The list of source terms"

    destination: Term = None
    "The destination term"

    input_edges_by_nodes: Dict[Term, List[Literal]] = dict()
    "The input literals by term"

    loops_by_nodes: Dict[Term, List[Literal]] = dict()
    "The loop literals by term"

    grounds: List[Literal] = dict()
    "The grounded terms"

    def __init__(self, sources, paths, grounds):
        """
        Creates a rule graph.

        :param sources: the list of source terms
        :type sources: List[Term]
        :param paths: the completed paths between the terms of the clause
        :type paths: List[RulePath]
        :param grounds: the remaining grounded literals in the clause
        :type grounds: List[Literal]
        """
        self.sources = sources
        self._build_graph(paths)
        self.grounds = grounds

    def _build_graph(self, paths):
        """
        Builds the graph.

        :param paths: the completed paths between the terms of the clause
        :type paths: List[RulePath]
        """
        for path in paths:
            for i in range(len(path.literals)):
                output_term = path.get_output_term(i)
                if path.is_loop(i):
                    literals = self.input_edges_by_nodes.setdefault(
                        output_term, [])
                else:
                    literals = self.loops_by_nodes.setdefault(output_term, [])
                literals.append(path.literals[i])