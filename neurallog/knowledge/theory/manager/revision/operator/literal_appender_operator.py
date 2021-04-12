"""
Handles some revision operators that append literals to the clause.
"""
import random
from abc import abstractmethod
from collections import Collection, deque
from typing import TypeVar, Generic, Set, Dict, List, Tuple

from neurallog.knowledge.examples import Examples, ExampleIterator, \
    ExamplesInferences, GroupedAtoms
from neurallog.knowledge.manager.tree_manager import TRUE_LITERAL
from neurallog.knowledge.theory.manager.revision.operator.revision_operator \
    import RevisionOperator, is_positive, relevant_breadth_first_search
from neurallog.language.equivalent_clauses import EquivalentAtom
from neurallog.language.language import HornClause, get_variable_atom, \
    Literal, Atom, Term, Number
from neurallog.util import OrderedSet
from neurallog.util.clause_utils import to_variable_atom
from neurallog.util.language import is_atom_unifiable_to_goal
from neurallog.util.multiprocessing.evaluation_transformer import \
    LiteralAppendAsyncTransformer, ConjunctionAppendAsyncTransformer
from neurallog.util.multiprocessing.multiprocessing import \
    DEFAULT_NUMBER_OF_PROCESS, \
    DEFAULT_EVALUATION_TIMEOUT, MultiprocessingEvaluation
from neurallog.util.multiprocessing.theory_evaluation import \
    AsyncTheoryEvaluator, \
    SyncTheoryEvaluator
from neurallog.util.variable_generator import VariableGenerator

V = TypeVar('V')

SUBSTITUTION_NAME = "__sub__"
TYPE_PREDICATE_NAME = "__type_sub__"


def create_substitution_map(head, answer):
    """
    Creates the substitution map to replace the constants from the relevant
    literals to their instantiated variable.

    :param head: the head of the query
    :type head: Atom
    :param answer: the answer of the example
    :type answer: Atom
    :return: the substitution map
    :rtype: Dict[Term, Term]
    """
    substitution_map: Dict[Term, Term] = dict()
    for i in range(head.arity()):
        substitution_map[answer.terms[i]] = head.terms[i]

    return substitution_map


def append_variable_atom_to_set(initial_atoms, variable_literals,
                                variable_map, variable_generator):
    """
    Appends the variable form of the `initial_atoms`, as literal, to the set
    of `variable_literals`.

    :param initial_atoms: the initial atoms
    :type initial_atoms: Set[Atom]
    :param variable_literals: the variable literals
    :type variable_literals: Set[Literal]
    :param variable_map: the variable map
    :type variable_map: Dict[Term, Term]
    :param variable_generator: the variable generator
    :type variable_generator: VariableGenerator
    """
    for atom in initial_atoms:
        variable_atom = to_variable_atom(atom, variable_generator, variable_map)
        variable_literals.add(Literal(variable_atom))


def build_all_literals_from_clause(initial_clause, candidates, answer_literals,
                                   skip_candidates, connected):
    """
    Creates the new candidate rules by adding on possible literal to the
    current rule's body. The current rule is represented by the head and body
    parameters.

    In addition, it skips equivalent sets, by checking if the free variables
    at the candidate atom can be renamed to match the free variables of a
    previously selected one. If an equivalent atom `A` is detected,
    the substitution map that makes it equals to another previous atom `B` is
    stored along side with `B`. In this case, when a rule from a set of
    candidates is selected for further refinements, it has a substitution map
    that, if applied to the candidates, can make them relevant to discarded
    atoms (like `A`), thus, it can be also considered relevant to `B`.

    :param initial_clause: the initial clause
    :type initial_clause: HornClause
    :param candidates: the candidates
    :type candidates: collections.Iterable[Literal]
    :param answer_literals: the set of candidate literals as the answer of
    the method
    :type answer_literals: Set[Literal]
    :param skip_candidates: the atoms to skip
    :type skip_candidates: Set[EquivalentAtom]
    :param connected: if `True`, only literals connected to the rules will be
    returned
    :type connected: bool
    """
    head = initial_clause.head
    fixed_terms: Set[Term] = set()
    for literal in initial_clause.body:
        fixed_terms.update(literal.terms)
    fixed_terms.update(head.terms)

    for candidate in candidates:
        if connected and fixed_terms.isdisjoint(candidate.terms):
            continue
        current_atom = EquivalentAtom(candidate, fixed_terms)
        if current_atom not in skip_candidates:
            skip_candidates.add(current_atom)
            answer_literals.add(candidate)


def build_substitution_clause(initial_clause, predicate_types):
    """
    Creates the clause to find the substitution of variables instantiated by the
    initial clause and a `type` clause, to associate the type of the
    substitution clause with the type of the `initial_clause`.

    :param initial_clause: the initial clause
    :type initial_clause: HornClause
    :param predicate_types: the program
    :type predicate_types: Dict[Predicate, Tuple[TermType]]
    :return: the clause to find the substitutions and the type clause
    :rtype: Tuple[HornClause, HornClause]
    """
    terms: List[Term] = []
    append_variables_from_atom(initial_clause.head, predicate_types, terms)

    body: List[Literal] = []
    if not initial_clause.body:
        body.append(TRUE_LITERAL)
    else:
        for literal in initial_clause.body:
            append_variables_from_atom(literal, predicate_types, terms)
        body = initial_clause.body

    substitution_head = Atom(SUBSTITUTION_NAME, *terms)
    substitution_clause = HornClause(substitution_head, *body)

    type_clause = HornClause(
        Atom(TYPE_PREDICATE_NAME, *terms),
        Literal(substitution_head), Literal(initial_clause.head))

    return substitution_clause, type_clause


def append_variables_from_atom(atom, predicate_types, append):
    """
    Appends the variables from the `atom` to the `append` list.

    :param atom: the atom
    :type atom: Atom
    :param predicate_types: the program
    :type predicate_types: Dict[Predicate, Tuple[TermType]]
    :param append: the append list
    :type append: List[Term]
    """
    term_types = predicate_types[atom.predicate]
    for i in range(atom.arity()):
        term = atom.terms[i]
        if not term.is_constant() and not term_types[i].number and \
                term not in append:
            append.append(term)


def build_queries_from_examples(examples, initial_clause_head,
                                substitution_head, positive_only):
    """
    Builds the queries from the positive examples to make possible find the
    substitution of each proved example.

    :param examples: the examples
    :type examples: Examples
    :param initial_clause_head: the initial clause head
    :type initial_clause_head: Atom
    :param substitution_head: the substitution clause head
    :type substitution_head: Atom
    :param positive_only: if `True`, only the positive examples will be
    considered
    :type positive_only: bool
    :return: the queries of the examples
    :rtype: Examples
    """
    query_set = Examples()
    predicate = initial_clause_head.predicate
    for example in ExampleIterator(examples, predicate=predicate):
        if not is_atom_unifiable_to_goal(example, initial_clause_head):
            continue
        if positive_only and example.weight <= 0.0:
            continue
        query = build_query_from_example(substitution_head, example)
        query_set.add_example(query)

    return query_set


def build_query_from_example(head, example):
    """
    Builds a query from the example, in order to find the substitution map
    for all variables in the initial clause to the constants that satisfies
    the example.

    :param head: the head of the initial clause
    :type head: Atom
    :param example: the example
    :type example: Atom
    :return: the query to find the substitutions
    :rtype: Atom
    """
    terms: List[Term] = list(example.terms)
    for i in range(len(terms), head.arity()):
        terms.append(head.terms[i])
    return Atom(SUBSTITUTION_NAME, *terms, weight=example.weight)


def get_initial_clause_head(initial_clause_head, target_predicate):
    """
    Gets the initial clause head, based on the initial clause and the
    target predicate.

    :param initial_clause_head: the head of the initial clause
    :type initial_clause_head: Atom
    :param target_predicate: the target predicate
    :type target_predicate: Predicate
    :return: the initial clause head
    :rtype: Atom
    """
    if target_predicate:
        return Atom(target_predicate, *initial_clause_head.terms)

    return initial_clause_head


class LiteralAppendOperator(RevisionOperator, Generic[V]):
    """
    Class of operators that append literals to a rule's body, given a set of
    examples, in order to improve the evaluation of the rule on those examples.
    """

    OPTIONAL_FIELDS = dict(RevisionOperator.OPTIONAL_FIELDS)
    OPTIONAL_FIELDS.update({
        "number_of_process": DEFAULT_NUMBER_OF_PROCESS,
        "evaluation_timeout": DEFAULT_EVALUATION_TIMEOUT,
        "relevant_depth": 0,
        "maximum_based_examples": -1,
        "positive_threshold": None,
        "infer_relevant": False
    })

    def __init__(self,
                 learning_system=None,
                 theory_metric=None,
                 clause_modifiers=None,
                 number_of_process=None,
                 evaluation_timeout=None,
                 relevant_depth=None,
                 maximum_based_examples=None,
                 positive_threshold=None,
                 infer_relevant=None
                 ):
        super().__init__(learning_system, theory_metric, clause_modifiers)

        self.number_of_process = number_of_process
        """
        The maximum number of process this class is allowed to create in order 
        to concurrently evaluate different rules. 

        The default is `1`.
        """
        if number_of_process is None:
            self.number_of_process = self.OPTIONAL_FIELDS["number_of_process"]

        self.evaluation_timeout = evaluation_timeout
        """
        The maximum amount of time, in seconds, allowed to the evaluation of 
        a rule.

        By default, it is 300 seconds, or 5 minutes. 
        """
        if evaluation_timeout is None:
            self.evaluation_timeout = self.OPTIONAL_FIELDS["evaluation_timeout"]

        self.relevant_depth = relevant_depth
        """
        The maximum depth on the transitivity of the relevant concept.
        An atom is relevant to the example if it shares (or transitively shares)
        a term with the example.

        If `relevant_depth` is `0`, it means that only the atoms which 
        actually share a term with the example will be considered, 
        these atoms are classed atoms at depth `0`.

        If it is `1`, it means that the atoms which share a term with the atoms 
        at depth `0` will also be considered.

        If it is `n`, for `n > 0`, it means that the atoms which share a term 
        with the atoms at depth `n - 1` will also be considered.

        If it is negative, atoms at any depth will be considered.  
        """
        if relevant_depth is None:
            self.relevant_depth = self.OPTIONAL_FIELDS["relevant_depth"]

        self.maximum_based_examples = maximum_based_examples
        "The maximum number of examples to base the revision."

        if maximum_based_examples is None:
            self.maximum_based_examples = \
                self.OPTIONAL_FIELDS["maximum_based_examples"]

        self.positive_threshold = positive_threshold
        """
        The threshold to consider an inferred atom as positive, if `None`, only 
        the atoms whose inferred value are greater than the `__null__` atom 
        are considered as positives. 
        """

        if self.positive_threshold is None:
            self.positive_threshold = self.OPTIONAL_FIELDS["positive_threshold"]

        self.infer_relevant = infer_relevant
        """
        If `True`, in addition to facts in the knowledge base, it also 
        considers as relevant the facts that could be inferred by the rules.
        """

        if self.infer_relevant is None:
            self.infer_relevant = self.OPTIONAL_FIELDS["infer_relevant"]

    # noinspection PyMissingOrEmptyDocstring
    def perform_operation(self, targets, minimum_threshold=None):
        theory = self.learning_system.theory.copy()
        initial_clause = self.build_empty_clause(targets)
        if not initial_clause:
            return theory

        horn_clause = self.build_extended_horn_clause(
            targets, initial_clause, set()).horn_clause
        horn_clause = self.apply_clause_modifiers(horn_clause, targets)
        theory.add_clauses(horn_clause)
        return theory

    # noinspection PyMethodMayBeStatic
    def build_empty_clause(self, targets):
        """
        Returns an initial empty clause, from the first target example in
        targets, by putting the head of the clause as the predicate of the
        examples, replacing the terms by new distinct variables.

        :param targets: the targets
        :type targets: Examples
        :return: the empty clause
        :rtype: HornClause
        """
        if not targets:
            return None

        for predicate in targets.keys():
            atom = get_variable_atom(predicate)
            return HornClause(atom)

    @abstractmethod
    def build_extended_horn_clause(
            self, examples, initial_clause, equivalent_literals,
            target_predicate=None):
        """
        Method to build an extended Horn clause that improves the metric on the
        examples, based on the initial clause.

        This method creates a Horn clause by adding a new literal to the body
        of the initial clause, if possible.

        This method should not modify the initial clause nor generate
        literals equivalent to the, possible empty, collection of equivalent
        literals.

        :param examples: the examples
        :type examples: Examples
        :param initial_clause: the initial clause
        :type initial_clause: HornClause
        :param equivalent_literals: the equivalent literals
        :type equivalent_literals: Set[Literal]
        :raise TheoryRevisionException: in case an error occur during the
        revision
        :param target_predicate: the target predicate, in case it is
        different from the the initial clause
        :type target_predicate: Optional[Predicate]
        :return: The extended horn clause
        :rtype: AsyncTheoryEvaluator or SyncTheoryEvaluator
        """
        pass

    def get_based_examples(self, examples):
        """
        Gets the examples on which to base the revision.

        :param examples: all possible examples
        :type examples: Examples
        :return: the examples to base the revision
        :rtype: Examples
        """
        positives = \
            list(filter(lambda x: is_positive(x), ExampleIterator(examples)))
        based_examples = Examples()
        if 0 < self.maximum_based_examples < len(positives):
            positives = random.sample(positives, self.maximum_based_examples)
        based_examples.add_examples(positives)
        return based_examples

    def get_literal_candidates_from_examples(
            self, initial_clause, substitution_goal, inferences,
            skip_candidates, connected):
        """
        Gets the literal candidates from the examples. The literals that are
        candidates to be appended to the initial clause in order to improve it.

        :param initial_clause: the initial clause
        :type initial_clause: HornClause
        :param substitution_goal: the substitution query
        :type substitution_goal: Atom
        :param inferences: the inferences of the `examples`
        :type inferences: ExamplesInferences
        :param skip_candidates: the candidates to skip
        :type skip_candidates: Set[EquivalentAtom]
        :param connected: if the returned literals must be connected to the
        rules.
        :type connected: bool
        :return: the candidate literals
        :rtype: Set[Literal]
        """
        variable_generator = VariableGenerator(
            map(lambda x: x.value, substitution_goal.terms))
        candidate_literals: Set[Literal] = set()
        grouped_atoms = \
            GroupedAtoms.group_examples(ExampleIterator(inferences.examples))
        for grouped_atom in grouped_atoms:
            constants = list(
                filter(lambda x: x.is_constant(), grouped_atom.query.terms))
            relevant_atoms = relevant_breadth_first_search(
                constants, self.relevant_depth, self.learning_system,
                infer_relevant=self.infer_relevant)
            variable_relevant: Set[Literal] = set()
            for answer in grouped_atom.atoms:
                substitution_map = \
                    create_substitution_map(substitution_goal, answer)
                append_variable_atom_to_set(
                    relevant_atoms, variable_relevant, substitution_map,
                    variable_generator)
                build_all_literals_from_clause(
                    initial_clause, variable_relevant, candidate_literals,
                    skip_candidates, connected)

        return candidate_literals


class RelevantLiteralAppendOperator(LiteralAppendOperator[Literal]):
    """
    A literal append operator that searches for a single literal, based on the
    relevant terms from the examples.
    """

    def __init__(self,
                 learning_system=None,
                 theory_metric=None,
                 clause_modifiers=None,
                 number_of_process=None,
                 evaluation_timeout=None,
                 relevant_depth=None,
                 maximum_based_examples=None
                 ):
        super().__init__(
            learning_system, theory_metric, clause_modifiers,
            number_of_process, evaluation_timeout, relevant_depth,
            maximum_based_examples)

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def initialize(self):
        super().initialize()
        self.literal_transformer = LiteralAppendAsyncTransformer(
            clause_modifiers=self.clause_modifiers)
        self.multiprocessing = MultiprocessingEvaluation(
            self.learning_system, self.theory_metric,
            self.literal_transformer,
            self.evaluation_timeout, self.number_of_process
        )

    # noinspection PyMissingOrEmptyDocstring,DuplicatedCode
    def build_extended_horn_clause(self, examples, initial_clause,
                                   equivalent_literals, target_predicate=None):
        initial_clause_head = \
            get_initial_clause_head(initial_clause.head, target_predicate)
        _initial_clause = HornClause(initial_clause_head, *initial_clause.body)
        predicate_types = dict(self.learning_system.knowledge_base.predicates)
        predicate_types.update(self.learning_system.theory.predicates)
        substitution_clause, type_clause = \
            build_substitution_clause(_initial_clause, predicate_types)
        query_set = build_queries_from_examples(
            self.get_based_examples(examples), initial_clause_head,
            substitution_clause.head, True)

        if not query_set:
            return None

        inferred_examples = \
            self.learning_system.infer_examples_appending_clauses(
                query_set, [substitution_clause, type_clause],
                positive_threshold=self.positive_threshold)
        skip_candidates = self.build_skip_candidates(
            initial_clause, equivalent_literals)
        literals = self.get_literal_candidates_from_examples(
            initial_clause, substitution_clause.head, inferred_examples,
            skip_candidates, True)
        if not literals:
            return None
        self.literal_transformer.initial_clause = initial_clause
        return self.multiprocessing.get_best_clause_from_candidates(
            literals, examples)

    @staticmethod
    def build_skip_candidates(initial_clause, equivalent_literals):
        """
        Builds the set of equivalent atoms to be skipped during the creation
        of the candidate literal. This set includes the literal that are
        already in the initial clause and the literals from the
        `equivalent_literals` collection.

        :param initial_clause: the initial clause
        :type initial_clause: HornClause
        :param equivalent_literals: the equivalent literal collection,
        i.e. the literal from other clause, to avoid creating equivalent clauses
        :type equivalent_literals: collections.Collection[Literal]
        :return: the set of equivalent atoms
        :rtype: Set[EquivalentAtom]
        """
        fixed_terms: Set[Term] = set()
        for literal in initial_clause.body:
            fixed_terms.update(literal.terms)
        fixed_terms.update(initial_clause.head.terms)
        skip_candidates: Set[EquivalentAtom] = set()
        for literal in initial_clause.body:
            skip_candidates.add(EquivalentAtom(literal, fixed_terms))
        for literal in equivalent_literals:
            skip_candidates.add(EquivalentAtom(literal, fixed_terms))
        return skip_candidates


class PathFinderAppendOperator(LiteralAppendOperator[Set[Literal]]):
    """
    A literal append operator that search for the literal based on the
    relevant terms from the examples and returns a path between the input and
    output terms.
    """

    OPTIONAL_FIELDS = dict(LiteralAppendOperator.OPTIONAL_FIELDS)
    OPTIONAL_FIELDS.update({
        "destination_index": -1,
        "maximum_path_length": -1
    })

    def __init__(self,
                 learning_system=None,
                 theory_metric=None,
                 clause_modifiers=None,
                 number_of_process=None,
                 evaluation_timeout=None,
                 relevant_depth=None,
                 maximum_based_examples=None,
                 destination_index=None,
                 maximum_path_length=None
                 ):
        super().__init__(
            learning_system, theory_metric, clause_modifiers,
            number_of_process, evaluation_timeout, relevant_depth,
            maximum_based_examples)
        self.destination_index = destination_index
        "The index of the term to be the destination of the path."

        if self.destination_index is None:
            self.destination_index = self.OPTIONAL_FIELDS["destination_index"]

        self.maximum_path_length = maximum_path_length
        "The maximum length of the path."

        if self.maximum_path_length is None:
            self.maximum_path_length = \
                self.OPTIONAL_FIELDS["maximum_path_length"]

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def initialize(self):
        super().initialize()
        self.conjunction_transformer: ConjunctionAppendAsyncTransformer = \
            ConjunctionAppendAsyncTransformer(
                clause_modifiers=self.clause_modifiers)
        self.multiprocessing = MultiprocessingEvaluation(
            self.learning_system, self.theory_metric,
            self.conjunction_transformer,
            self.evaluation_timeout, self.number_of_process
        )

    # noinspection PyMissingOrEmptyDocstring,DuplicatedCode
    def build_extended_horn_clause(self, examples, initial_clause,
                                   equivalent_literals, target_predicate=None):
        predicate_types = dict(self.learning_system.knowledge_base.predicates)
        predicate_types.update(self.learning_system.theory.predicates)
        substitution_clause, type_clause = build_substitution_clause(
            initial_clause, predicate_types)
        initial_clause_head = \
            get_initial_clause_head(initial_clause.head, target_predicate)
        query_set = build_queries_from_examples(
            self.get_based_examples(examples), initial_clause_head,
            substitution_clause.head, True)

        if not query_set:
            return None

        inferred_examples = \
            self.learning_system.infer_examples_appending_clauses(
                query_set, [substitution_clause, type_clause],
                positive_threshold=self.positive_threshold)
        literals = self.get_literal_candidates_from_examples(
            initial_clause, substitution_clause.head, inferred_examples,
            set(), False)
        if not literals:
            return None
        path_finder = PathFinder(literals)
        # TODO: change it to find all paths
        paths: Collection[Tuple[Term]] = path_finder.shortest_path(
            initial_clause_head.terms[0],
            initial_clause_head.terms[self.destination_index],
            self.maximum_path_length)
        if not paths:
            return None
        conjunctions: Set[OrderedSet[Term]] = set()
        for path in paths:
            path_finder.path_to_rules(path, conjunctions)
        self.conjunction_transformer.initial_clause = initial_clause
        return self.multiprocessing.get_best_clause_from_candidates(
            conjunctions, examples)


class PathFinder:
    """
    Class to find paths in a set of atoms.
    """

    def __init__(self, atoms):
        self.atoms = atoms
        self.constants = set()
        self._cached_atoms_by_term: Dict[Term, Set[Atom]] = dict()
        for atom in self.atoms:
            for term in atom:
                if isinstance(term, Number):
                    continue
                self.constants.add(term)
                self._cached_atoms_by_term.setdefault(term, set()).add(atom)

    def get_neighbour_terms(self, term):
        """
        Gets the neighbour terms of the `term`.

        :param term: the term
        :type term: Term
        :return: the neighbour terms
        :rtype: Collection[Term]
        """
        if isinstance(term, Number):
            return set()

        neighbours = set()
        for atom in self.get_atoms_with_term(term):
            for neighbour in atom.terms:
                if term != neighbour and not isinstance(neighbour, Number):
                    neighbours.add(neighbour)

        return neighbours

    def get_atoms_with_term(self, term):
        """
        Gets the atoms with the term.

        :param term: the term
        :type term: Term
        :return: the set of atom containing the term
        :rtype: Set[Atom]
        """
        return self._cached_atoms_by_term.get(term, set())

    def shortest_path(self, source, destination, maximum_length):
        """
        Finds the shortest paths (sequence of terms), of at most
        `maximum_length` long, between the `source` and the `destination`
        terms in the knowledge base. If such path exists.

        :param source: the source term
        :type source: Term
        :param destination: the destination term
        :type destination: Term
        :param maximum_length: the maximum length of the path. If negative,
        there will be no limit on the length of the path.
        :type maximum_length: int
        :return: the shortest paths between `source` and `destination`
        :rtype: Collection[Tuple[Term]]
        """
        if not self.constants.issuperset({source, destination}):
            return None
        if source == destination or \
                destination in self.get_neighbour_terms(source):
            return [(source, destination)]

        return self._find_shortest_paths(source, destination, maximum_length)

    def _find_shortest_paths(self, source, destination, maximum_length):
        """
        Finds the shortest paths (sequence of terms), of at most
        `maximum_length` long, between the `source` and the
        `destination`
        terms in the knowledge base. If such path exists.

        :param source: the source term
        :type source: Term
        :param destination: the destination term
        :type destination: Term
        :param maximum_length: the maximum length of the path. If
        negative,
        there will be no limit on the length of the path.
        :type maximum_length: int
        :return: the shortest paths between `source` and `destination`
        :rtype: Collection[Tuple[Term]]
        """
        distance_for_vertex: Dict[Term, int] = dict()
        predecessors_of_vertex: Dict[Term, Set[Term]] = dict()

        distance_for_vertex[source] = 0

        queue: deque[Term] = deque()
        queue.append(source)
        found = False
        previous_distance = 0
        while queue:
            current = queue.popleft()
            distance = distance_for_vertex[current]
            if found and distance > previous_distance:
                break
            previous_distance = distance
            if 0 <= maximum_length < distance + 1:
                break

            for neighbor in self.get_neighbour_terms(current):
                predecessors_of_vertex.setdefault(neighbor, set()).add(current)
                if neighbor not in distance_for_vertex:
                    distance_for_vertex[neighbor] = distance + 1
                    queue.append(neighbor)

                if neighbor == destination:
                    found = True
                    break

        if not found:
            return set()
        return self._build_paths(
            source, destination, predecessors_of_vertex,
            distance_for_vertex[destination])

    @staticmethod
    def _build_paths(source, destination, predecessors_of_vertex, path_length):
        """
        Builds the paths based on the predecessors. Then, filters it by the
        ones that starts on `source` and ends at `destination`.

        :param source: the source of the path
        :type source: Term
        :param destination: the destination of the path
        :type destination: Term
        :param predecessors_of_vertex:
        :type predecessors_of_vertex: Dict[Term, Set[Term]]
        :param path_length: the length of the paths
        :type path_length: int
        :return: the paths that starts on `source` and ends at `destination`
        :rtype: Collection[Tuple[Term]]
        """
        queue: deque[List[Term]] = deque()
        queue.append([destination])

        for i in range(path_length - 1, -1, -1):
            size = len(queue)
            for j in range(size):
                current_array = queue.popleft()
                for predecessor in predecessors_of_vertex[current_array[0]]:
                    queue.append([predecessor] + list(current_array))

        result = \
            filter(lambda x: x[0] == source and x[-1] == destination, queue)
        return set(map(lambda x: tuple(x), result))

    def path_to_rules(self, path, append):
        """
        Creates rules where the body is the path between terms in a
        knowledge base.

        :param path: the path
        :type path: Tuple[Term] or List[Term]
        :param append: the collection of rules to append
        :type append: Set[Iterable[Literal]]
        :return: the collection of rules
        :rtype: Set[Iterable[Literal]]
        """
        current_edges = set(self.get_atoms_with_term(path[0]))
        path_length = len(path) - 1
        queue: deque[OrderedSet[Literal]] = deque()
        queue.append(OrderedSet())
        for i in range(path_length):
            size = len(queue)
            next_edges = set(self.get_atoms_with_term(path[i + 1]))
            current_edges = current_edges.intersection(next_edges)
            for j in range(size):
                current_array = queue.popleft()
                for edge in current_edges:
                    auxiliary = OrderedSet(current_array)
                    auxiliary.add(Literal(edge))
                    queue.append(auxiliary)
            current_edges = next_edges

        append.update(queue)

        return append
