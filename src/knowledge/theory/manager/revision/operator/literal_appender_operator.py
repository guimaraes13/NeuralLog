"""
Handles some revision operators that append literals to the clause.
"""
from abc import abstractmethod
from collections import Collection, deque
from random import random
from typing import TypeVar, Generic, Set, Dict, List, Tuple

from src.knowledge.examples import Examples, ExampleIterator, ExamplesInferences
from src.knowledge.program import NeuralLogProgram
from src.knowledge.theory.manager.revision.operator.revision_operator import \
    RevisionOperator, is_positive, relevant_breadth_first_search
from src.language.equivalent_clauses import EquivalentAtom
from src.language.language import HornClause, get_variable_atom, Literal, \
    Atom, \
    Term
from src.util import OrderedSet
from src.util.clause_utils import to_variable_atom
from src.util.language import is_atom_unifiable_to_goal
from src.util.multiprocessing.evaluation_transformer import \
    LiteralAppendAsyncTransformer, ConjunctionAppendAsyncTransformer
from src.util.multiprocessing.multiprocessing import \
    DEFAULT_NUMBER_OF_PROCESS, \
    DEFAULT_EVALUATION_TIMEOUT, MultiprocessingEvaluation
from src.util.multiprocessing.theory_evaluation import AsyncTheoryEvaluator, \
    SyncTheoryEvaluator
from src.util.variable_generator import VariableGenerator

V = TypeVar('V')

SUBSTITUTION_NAME = "__sub__"


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
        variable_atom = to_variable_atom(atom, variable_map, variable_generator)
        variable_literals.add(Literal(variable_atom))
        pass


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


def build_substitution_clause(initial_clause):
    """
    Creates the clause to find the substitution of variables instantiated by the
    initial clause.

    :param initial_clause: the initial clause
    :type initial_clause: HornClause
    :return: the clause to find the substitutions
    :rtype: HornClause
    """
    terms: List[Term] = []
    append_variables_from_atom(initial_clause.head, terms)

    body: List[Literal] = []
    if not initial_clause.body:
        body.append(NeuralLogProgram.TRUE_ATOM)
    else:
        for literal in initial_clause.body:
            append_variables_from_atom(literal, terms)
        body = initial_clause.body

    return HornClause(Atom(SUBSTITUTION_NAME, *terms), *body)


def append_variables_from_atom(atom, append):
    """
    Appends the variables from the `atom` to the `append` list.

    :param atom: the atom
    :type atom: Atom
    :param append: the append list
    :type append: List[Term]
    """
    for term in atom.terms:
        if not term.is_constant() and term not in append:
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
        query_set.add_example(
            build_query_from_example(substitution_head, example))

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
    terms: List[Term] = [] + example.terms
    for i in range(len(example.terms), head.arity()):
        terms.append(head.terms[i])
    return Atom(SUBSTITUTION_NAME, *terms, weight=example.weight)


class LiteralAppendOperator(RevisionOperator, Generic[V]):
    """
    Class of operators that append literals to a rule's body, given a set of
    examples, in order to improve the evaluation of the rule on those examples.
    """

    OPTIONAL_FIELDS = super().OPTIONAL_FIELDS
    OPTIONAL_FIELDS.update({
        "modifier_clause_before_evaluate": False,
        "number_of_process": DEFAULT_NUMBER_OF_PROCESS,
        "evaluation_timeout": DEFAULT_EVALUATION_TIMEOUT,
        "relevant_depth": 0,
        "maximum_based_examples": -1
    })

    def __init__(self,
                 learning_system=None,
                 theory_metric=None,
                 clause_modifiers=None,
                 modifier_clause_before_evaluate=None,
                 number_of_process=None,
                 evaluation_timeout=None,
                 relevant_depth=None,
                 maximum_based_examples=None
                 ):
        super().__init__(learning_system, theory_metric, clause_modifiers)

        self.modifier_clause_before_evaluate = modifier_clause_before_evaluate
        """
        If `True`, apply the clause modifier to the clause before 
        evaluating the clause to decide which one is the best.
        """

        if modifier_clause_before_evaluate is None:
            self.modifier_clause_before_evaluate = \
                self.OPTIONAL_FIELDS["modifier_clause_before_evaluate"]

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

    # noinspection PyMissingOrEmptyDocstring
    def perform_operation(self, targets):
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
            self, examples, initial_clause, equivalent_literals):
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
        based_examples.add_all(positives)
        return based_examples

    def get_literal_candidates_from_examples(
            self, initial_clause, substitution_goal, inferences,
            skip_candidates, connected):
        """
        Gets the literal candidates from the examples. The literals that are
        candidates to be appended to the initial clause in order to get
        improve it.

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
        for predicate, inferred in inferences.items():
            for key, value in inferred.items():
                example = inferences.examples.get(predicate, dict()).get(key)
                if not example:
                    continue
                constants = list(
                    filter(lambda x: x.is_constant(), example.terms))
                relevant_atoms = relevant_breadth_first_search(
                    constants, self.relevant_depth, self.learning_system)
                variable_relevant: Set[Literal] = set()
                substitution_map = create_substitution_map(
                    substitution_goal, example)
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
                 modifier_clause_before_evaluate=None,
                 number_of_process=None,
                 evaluation_timeout=None,
                 relevant_depth=None,
                 maximum_based_examples=None
                 ):
        super().__init__(
            learning_system, theory_metric, clause_modifiers,
            modifier_clause_before_evaluate, number_of_process,
            evaluation_timeout, relevant_depth, maximum_based_examples)

    # noinspection PyMissingOrEmptyDocstring,PyAttributeOutsideInit
    def initialize(self):
        super().initialize()
        self.literal_transformer = \
            LiteralAppendAsyncTransformer(self.clause_modifiers)
        self.multiprocessing = MultiprocessingEvaluation(
            self.learning_system, self.theory_metric,
            self.literal_transformer,
            self.evaluation_timeout, self.number_of_process
        )

    # noinspection PyMissingOrEmptyDocstring
    def build_extended_horn_clause(self, examples, initial_clause,
                                   equivalent_literals):
        substitution_clause = build_substitution_clause(initial_clause)
        query_set = build_queries_from_examples(
            self.get_based_examples(examples), initial_clause.head,
            substitution_clause.head, True)

        if not query_set:
            return None
        inferred_examples = \
            self.learning_system.infer_examples_appending_clauses(
                query_set, [substitution_clause])
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

    OPTIONAL_FIELDS = super().OPTIONAL_FIELDS
    OPTIONAL_FIELDS.update({
        "destination_index": -1,
        "maximum_path_length": -1
    })

    def __init__(self,
                 learning_system=None,
                 theory_metric=None,
                 clause_modifiers=None,
                 modifier_clause_before_evaluate=None,
                 number_of_process=None,
                 evaluation_timeout=None,
                 relevant_depth=None,
                 maximum_based_examples=None,
                 destination_index=None,
                 maximum_path_length=None
                 ):
        super().__init__(
            learning_system, theory_metric, clause_modifiers,
            modifier_clause_before_evaluate, number_of_process,
            evaluation_timeout, relevant_depth, maximum_based_examples)
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
            ConjunctionAppendAsyncTransformer(self.clause_modifiers)
        self.multiprocessing = MultiprocessingEvaluation(
            self.learning_system, self.theory_metric,
            self.conjunction_transformer,
            self.evaluation_timeout, self.number_of_process
        )

    # noinspection PyMissingOrEmptyDocstring
    def build_extended_horn_clause(self, examples, initial_clause,
                                   equivalent_literals):
        substitution_clause = build_substitution_clause(initial_clause)
        head = initial_clause.head
        query_set = build_queries_from_examples(
            self.get_based_examples(examples), head, substitution_clause.head,
            True)
        if not query_set:
            return None
        inferred_examples = \
            self.learning_system.infer_examples_appending_clauses(
                query_set, [substitution_clause])
        literals = self.get_literal_candidates_from_examples(
            initial_clause, substitution_clause.head, inferred_examples,
            set(), False)
        if not literals:
            return None
        knowledge_base = NeuralLogProgram()
        for literal in literals:
            knowledge_base.add_fact(literal)
        knowledge_base.build_program()
        paths: Collection[Tuple[Term]] = knowledge_base.shortest_path(
            head.terms[0], head.terms[self.destination_index],
            self.maximum_path_length)
        if not paths:
            return None
        conjunctions: Set[OrderedSet[Term]] = set()
        for path in paths:
            self.path_to_rules(path, knowledge_base, conjunctions)
        self.conjunction_transformer.initial_clause = initial_clause
        return self.multiprocessing.get_best_clause_from_candidates(
            conjunctions, examples)

    @staticmethod
    def path_to_rules(path, knowledge_base, append):
        """
        Creates rules where the body is the path between terms in a
        knowledge base.

        :param path: the path
        :type path: Tuple[Term] or List[Term]
        :param knowledge_base: the knowledge base
        :type knowledge_base: NeuralLogProgram
        :param append: the collection of rules to append
        :type append: Set[Iterable[Literal]]
        :return: the collection of rules
        :rtype: Set[Iterable[Literal]]
        """
        current_edges = set(knowledge_base.get_atoms_with_term(path[0]))
        path_length = len(path) - 1
        queue: deque[OrderedSet[Literal]] = deque()
        queue.append(OrderedSet())
        for i in range(path_length):
            size = len(queue)
            next_edges = set(knowledge_base.get_atoms_with_term(path[i + 1]))
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
