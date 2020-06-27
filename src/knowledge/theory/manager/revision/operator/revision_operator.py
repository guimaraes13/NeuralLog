"""
Handle the revision operators.
"""
import itertools
import logging
from abc import abstractmethod
from collections import Collection, deque
from typing import Dict, Set, List, Deque, TypeVar, Any

import src.structure_learning.structure_learning_system as sls
from src.knowledge.examples import Examples, ExampleIterator, ExamplesInferences
from src.knowledge.program import NeuralLogProgram
from src.knowledge.theory import TheoryRevisionException
from src.knowledge.theory.evaluation.metric.theory_metric import TheoryMetric
from src.knowledge.theory.evaluation.theory_evaluator import TheoryEvaluator
from src.knowledge.theory.manager.revision.clause_modifier import ClauseModifier
from src.language.equivalent_clauses import EquivalentClauseAtom, \
    EquivalentHornClause
from src.language.language import KnowledgeException, Atom, Predicate, \
    HornClause, Term, Number, Literal, ClauseProvenance
from src.util import Initializable, InitializationException, OrderedSet
from src.util.clause_utils import apply_substitution, to_variable_atom, \
    may_rule_be_safe, get_non_negated_literals_with_head_variable, \
    is_rule_safe, \
    find_biggest_gap
from src.util.language import iterable_to_string
from src.util.multiprocessing.evaluation_transformer import \
    EquivalentHonClauseAsyncTransformer
from src.util.multiprocessing.multiprocessing import MultiprocessingEvaluation
from src.util.multiprocessing.theory_evaluation import AsyncTheoryEvaluator
from src.util.variable_generator import VariableGenerator

DEFAULT_VARIABLE_GENERATOR = VariableGenerator

logger = logging.getLogger(__name__)

cached_null_atoms: Dict[Predicate, Atom] = dict()


def relevant_breadth_first_search(terms, relevant_depth,
                                  learning_system, safe_stop=False):
    """
    Retrieve the relevant atom, given the `terms`, by performing a
    breadth-first search in the knowledge base graph, until a given
    `relevant_depth`.

    :param terms: the terms
    :type terms: Collection[Term]
    :param relevant_depth: the distance maximum to the initial term,
    to be considered as relevant; if negative, it considers all found term as
    relevant
    :type relevant_depth: int
    :param learning_system: the learning system
    :type learning_system: sls.StructureLearningSystem
    :param safe_stop: if `True`, the search stops when all the atoms of a
    distance are added and those atoms, collectively, contains all `terms`
    :type safe_stop: bool
    :return: the set of relevant atoms with respect to the `terms`
    :rtype: Set[Atom]
    """
    terms_distance: Dict[Term, int] = dict()
    queue = deque()
    atoms: Set[Atom] = set()
    current_relevant: Set[Term] = set()
    head_terms: Set[Term] = set()
    body_terms: Set[Term] = set()

    for term in terms:
        if isinstance(term, Number):
            continue
        terms_distance[term] = 0
        queue.append(term)
        current_relevant.add(term)
        head_terms.add(term)

    atom_set = learning_system.inferred_relevant(current_relevant)
    atoms.update(atom_set)

    previous_distance = 0
    while queue:
        current_term = queue.popleft()
        current_distance = terms_distance[current_term]

        if current_distance != previous_distance:
            atom_set = learning_system.inferred_relevant(current_relevant)
            atoms.update(atom_set)
            if safe_stop:
                # If `safe_stop`, the minimal safe rule (i.e. the rule where
                # all atoms in the head appears in the body) is returned.
                # So, there is no point in adding more atoms beyond that.
                # The body_terms contains the already found terms, so it can
                # check if this method can stop early, by this criteria
                for atom in atom_set:
                    body_terms.update(atom.terms)
                if body_terms.issuperset(head_terms):
                    # The rule is safe, can return the atoms
                    break
            current_relevant = set()
            previous_distance = current_distance

        if current_term not in terms_distance:
            current_relevant.add(current_term)

        atom_set = \
            learning_system.knowledge_base.get_atoms_with_term(current_term)
        atoms.update(atom_set)

        if relevant_depth < 0 or current_distance < relevant_depth:
            neighbour_terms = \
                learning_system.knowledge_base.get_neighbour_terms(current_term)
            for neighbour in neighbour_terms:
                if neighbour not in terms_distance:
                    terms_distance[neighbour] = current_distance + 1
                    queue.append(neighbour)

    return atoms


def get_null_example(knowledge_base, predicate):
    """
    Gets and cached the null example for the predicate.

    :param knowledge_base: the knowledge base
    :type knowledge_base: NeuralLogProgram
    :param predicate: the predicate
    :type predicate: Predicate
    :return: the null example
    :rtype: Atom
    """
    null_atom = cached_null_atoms.get(predicate)
    if null_atom is None:
        null_atom = sls.build_null_atom(knowledge_base, predicate)
        cached_null_atoms[predicate] = null_atom

    return null_atom


def is_covered(knowledge_base, example, inferred_examples):
    """
    Checks if the example is covered.

    :param knowledge_base: the knowledge base
    :type knowledge_base: NeuralLogProgram
    :param example: the example
    :type example: Atom
    :param inferred_examples: the inference of the examples
    :type inferred_examples: ExamplesInferences
    :return: `True`, if the example is covered; otherwise, `False`
    :rtype: bool
    """
    null_atom = get_null_example(knowledge_base, example.predicate)
    null_value = inferred_examples.get_value_for_example(null_atom)
    example_value = inferred_examples.get_value_for_example(example)

    return example_value > null_value


def build_minimal_safe_equivalent_clauses(bottom_clause):
    """
    Builds a set of Horn clauses with the least number of literals in the
    body that makes the clause safe.

    :param bottom_clause: the bottom clause
    :type bottom_clause: HornClause
    :raise TheoryRevisionException: if an error occurs during the revision
    :return: a set of Horn clauses, where each clause contains the minimal
    set of literals, in its body, which makes the clause safe.
    :rtype: Set[EquivalentHornClause] or None
    """

    if not may_rule_be_safe(bottom_clause):
        raise TheoryRevisionException(
            "Error when generating a new rule, the generate rule can not "
            "become safe.")

    candidate_literals = \
        list(get_non_negated_literals_with_head_variable(bottom_clause))
    candidate_literals.sort(key=lambda x: str(x.predicate))
    queue = deque()
    queue.append(EquivalentHornClause(bottom_clause.head))
    for _ in range(len(candidate_literals)):
        append_all_candidates_to_queue(queue, candidate_literals)
        safe_clauses = set(filter(lambda x: is_rule_safe(x.horn_clause), queue))
        if safe_clauses:
            return safe_clauses

    return None


def append_all_candidates_to_queue(queue, candidates):
    """
    Creates a list of equivalent Horn clauses containing a equivalent Horn
    clause for each substitution of each candidate, skipping equivalent
    clauses.

    :param queue: the queue with initial clauses
    :type queue: deque[EquivalentHornClause]
    :param candidates: the list of candidates
    :type candidates: List[Literal]
    """
    skip_atom: Dict[EquivalentClauseAtom, EquivalentClauseAtom] = dict()
    skip_clause: Dict[EquivalentClauseAtom, EquivalentHornClause] = dict()

    size = len(queue)  # the initial size of the queue
    for _ in range(size):
        equivalent_horn_clause = queue.popleft()
        queue.extend(equivalent_horn_clause.build_initial_clause_candidates(
            candidates, skip_atom, skip_clause))


def remove_equivalent_candidates(candidates, equivalent_clause):
    """
    Removes all equivalent candidates of the body of the clause from the
    candidate set.

    :param candidates: the candidate set
    :type candidates: Set[Literal]
    :param equivalent_clause: the equivalent clause
    :type equivalent_clause: EquivalentHornClause
    """
    for literal in equivalent_clause.clause_body:
        for substitutions in equivalent_clause.substitution_maps:
            candidates.discard(apply_substitution(literal, substitutions))
        candidates.discard(literal)


def remove_last_literal_equivalent_candidates(candidates, equivalent_clause):
    """
    Removes all equivalent candidates of the body of the clause from the
    candidate set.

    :param candidates: the candidate set
    :type candidates: Set[Literal]
    :param equivalent_clause: the equivalent clause
    :type equivalent_clause: EquivalentHornClause
    """
    for substitutions in equivalent_clause.substitution_maps:
        candidates.discard(apply_substitution(
            equivalent_clause.last_literal, substitutions))
    candidates.discard(equivalent_clause.last_literal)


def is_positive(example):
    """
    Check if the example is positive.

    :param example: the example
    :type example: Atom
    :return: `True` if the example is positive; otherwise, `False`
    :rtype: bool
    """
    return example.weight > 0.0


def add_variable_substitutions(atom, variable_map, variables):
    """
    Adds the substitution of the i-th term of the atom as the i-th term from
    the variable list, if it is a constant, to the `variable_map`.

    :param atom: the atom
    :type atom: Atom
    :param variable_map: the variable map
    :type variable_map: Dict[Term, Term]
    :param variables: the examples' variables
    :type variables: List[Term]
    """
    for i in range(len(atom.terms)):
        if atom.terms[i].is_constant() and \
                not atom.terms[i] not in variable_map:
            variable_map[atom.terms[i]] = variables[i]


T = TypeVar('T')


# noinspection PyMissingOrEmptyDocstring
def my_cmp_to_key(my_cmp, key):
    """
    Convert a `my_cmp` function into a `key` function.

    :param my_cmp: the comparator function
    :type my_cmp: Callable[[T, T], int]
    :param key: a function to extract the key from the object
    :type key: Callable[[Any], T]
    :return: a class to represent the object
    :rtype: Callable
    """

    class K(object):
        __slots__ = ['obj']

        def __init__(self, obj):
            self.obj = obj

        def __lt__(self, other):
            return my_cmp(key(self.obj), key(other.obj)) < 0

        def __gt__(self, other):
            return my_cmp(key(self.obj), key(other.obj)) > 0

        def __eq__(self, other):
            return my_cmp(key(self.obj), key(other.obj)) == 0

        def __le__(self, other):
            return my_cmp(key(self.obj), key(other.obj)) <= 0

        def __ge__(self, other):
            return my_cmp(key(self.obj), key(other.obj)) >= 0

        __hash__ = None

    return K


class LearnedClause(ClauseProvenance):
    """
    Represents the provenance of a clause learned by a revision operator.
    """

    def __init__(self, creator):
        self.creator = creator
        self.modifiers = []

    def add_modifier(self, modifier):
        """
        Adds the `modifier` to the list of modifiers.

        :param modifier: the modifier
        :type modifier: Any
        """
        self.modifiers.append(modifier)

    def __repr__(self):
        message = "create by {}".format(self.creator)
        if self.modifiers:
            message += " modifier by: {}".format(", ".join(self.modifiers))
        return message


class RevisionOperator(Initializable):
    """
    Operator to revise the theory.
    """

    OPTIONAL_FIELDS: Dict[str, Any] = {"clause_modifiers": None}

    def __init__(self, learning_system=None, theory_metric=None,
                 clause_modifiers=None):
        """
        Creates a revision operator.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param theory_metric: the theory metric
        :type theory_metric: TheoryMetric
        :param clause_modifiers: a clause modifier, a list of clause modifiers
        or none
        :type clause_modifiers: ClauseModifier or Collection[ClauseModifier]
        or None
        """
        self.learning_system = learning_system
        self.theory_metric = theory_metric
        self.clause_modifiers: Collection[ClauseModifier] = clause_modifiers

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        if self.clause_modifiers is None:
            self.clause_modifiers = []
        elif not isinstance(self.clause_modifiers, Collection):
            self.clause_modifiers = [self.clause_modifiers]
        for clause_modifier in self.clause_modifiers:
            clause_modifier.learning_system = self.learning_system
            clause_modifier.initialize()
        self.theory_metric.initialize()

    # noinspection PyMissingOrEmptyDocstring
    @property
    def theory_evaluator(self) -> TheoryEvaluator:
        return self.learning_system.theory_evaluator

    # noinspection PyMissingOrEmptyDocstring
    def required_fields(self):
        return ["learning_system", "theory_metric"]

    @abstractmethod
    def perform_operation(self, targets):
        """
        Applies the operation on the theory, given the target examples.

        :param targets: the target examples
        :type targets: Examples
        :return: the revised theory
        :rtype: NeuralLogProgram or None
        """
        pass

    @abstractmethod
    def theory_revision_accepted(self, revised_theory):
        """
        Method to send a feedback to the revision operator, telling
        that the
        revision was accepted.

        :param revised_theory: the revised theory
        :type revised_theory: NeuralLogProgram
        """
        pass

    def apply_clause_modifiers(self, horn_clause, targets):
        """
        Applies the clause modifiers to the `horn_clause`, given the target
        examples.

        :param horn_clause: the Horn clause
        :type horn_clause: HornClause
        :param targets: the target examples
        :type targets: Examples
        :return: the modified Horn clause
        :rtype: HornClause
        """
        for clause_modifier in self.clause_modifiers:
            horn_clause = clause_modifier.modify_clause(horn_clause, targets)
        return horn_clause

    def __repr__(self):
        return self.__class__.__name__


class BottomClauseBoundedRule(RevisionOperator):
    """
    Operator that implements Guimarães and Paes rule creation algorithm.

    V. Guimarães and A. Paes, Looking at the Bottom and the Top: A Hybrid
    Logical Relational Learning System Based on Answer Sets, 2015 Brazilian
    Conference on Intelligent Systems (BRACIS), Natal, 2015, pp. 240-245.
    """

    OPTIONAL_FIELDS = RevisionOperator.OPTIONAL_FIELDS
    OPTIONAL_FIELDS.update({
        "variable_generator": None,
        "relevant_depth": 0,
        "refine": False,
        "maximum_side_way_movements": 0,
        "improvement_threshold": 0.0,
        "generic": True,
        "evaluation_timeout": 300,
        "number_of_process": 1
    })

    def __init__(self,
                 learning_system=None,
                 theory_metric=None,
                 clause_modifiers=None,
                 variable_generator=None,
                 relevant_depth=None,
                 refine=None,
                 maximum_side_way_movements=None,
                 improvement_threshold=None,
                 generic=None,
                 evaluation_timeout=None,
                 number_of_process=None):
        """
        Creates a Bottom Clause Bounded Rule operator.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param theory_metric: the theory metric
        :type theory_metric: TheoryMetric
        :param clause_modifiers: a clause modifier, a list of clause modifiers
        or none
        :type clause_modifiers: ClauseModifier or Collection[ClauseModifier]
        or None
        :param variable_generator: the variable generator
        :type variable_generator: Optional[VariableGenerator]
        :param relevant_depth: the relevant depth
        :type relevant_depth: Optional[int]
        :param refine: if it is to refine the rules
        :type refine: Optional[bool]
        :param maximum_side_way_movements: the maximum side way movements
        :type maximum_side_way_movements: Optional[int]
        :param improvement_threshold: the improvement threshold
        :type improvement_threshold: Optional[float]
        :param generic: if it is to return the most generic rule
        :type generic: Optional[bool]
        :param evaluation_timeout: the evaluation timeout, in seconds
        :type evaluation_timeout: Optional[int]
        :param number_of_process: the number of parallel process
        :type number_of_process: Optional[int]
        """
        super().__init__(learning_system, theory_metric, clause_modifiers)

        self.variable_generator = variable_generator
        "The variable name generator."

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

        self.refine = refine
        """
        It specifies if the rule must be reined by adding literals to it,
        in order to try to improve the rule.
        """
        if refine is None:
            self.refine = self.OPTIONAL_FIELDS["refine"]

        self.maximum_side_way_movements = maximum_side_way_movements
        """
        The maximum side way movements, this is, the maximum number of 
        refining steps will be made, without improving the performance.
        
        If a metric improves by adding a literal to its body, it does not 
        count as a side way movement and the number of side way steps at the 
        moment becomes zero.
        
        If it is negative, there will be no maximum side way movements, 
        wall possible literals will be tried, since it does not degrade the 
        rule.
        """
        if maximum_side_way_movements is None:
            self.maximum_side_way_movements = \
                self.OPTIONAL_FIELDS["maximum_side_way_movements"]

        self.improvement_threshold = improvement_threshold
        """
        The minimal necessary improvements, over the current clause evaluation 
        and a new candidate, to be considered as improvement. If the threshold
        is not met, it is considered a side way movement.
        
        Use a threshold of `0.0` and a negative `maximum_side_way_movements` 
        to allow the search to test all possible rules.
        
        Use a threshold of `e` and a `maximum_side_way_movements` of `0` to 
        stop as soon as a rule does not improve more than `e`. 
        """
        if improvement_threshold is None:
            self.improvement_threshold = \
                self.OPTIONAL_FIELDS["improvement_threshold"]

        self.generic = generic
        """
        Flag to specify which rule will be returned in case of a tie in the 
        evaluation of the best rules.
        
        If `generic` is `True`, the most generic tied rule will be returned, 
        this is, the rule whose body has the fewest number of literals in it.
        
        if `generic` is `False`, the most specific rule will be returned, 
        instead, this is, the rule whose body has the most number of literals 
        in it. 
        """
        if generic is None:
            self.generic = self.OPTIONAL_FIELDS["generic"]

        self.evaluation_timeout = evaluation_timeout
        """
        The maximum amount of time, in seconds, allowed to the evaluation of 
        a rule.
        
        By default, it is 300 seconds, or 5 minutes. 
        """
        if evaluation_timeout is None:
            self.evaluation_timeout = self.OPTIONAL_FIELDS["evaluation_timeout"]

        self.number_of_process = number_of_process
        """
        The maximum number of process this class is allowed to create in order 
        to concurrently evaluate different rules. 
        
        The default is `1`.
        """
        if number_of_process is None:
            self.number_of_process = self.OPTIONAL_FIELDS["number_of_process"]

    # noinspection PyMissingOrEmptyDocstring
    def initialize(self):
        super().initialize()
        if self.variable_generator is None:
            self.variable_generator = DEFAULT_VARIABLE_GENERATOR()
        # noinspection PyAttributeOutsideInit
        self.multiprocessing = MultiprocessingEvaluation(
            self.learning_system, self.theory_metric,
            EquivalentHonClauseAsyncTransformer(),
            self.evaluation_timeout, self.number_of_process
        )

    # noinspection PyMissingOrEmptyDocstring
    def perform_operation(self, targets):
        try:
            logger.info("Performing operation on\t%d examples.", targets.size())
            theory = self.learning_system.theory.copy()
            inferred_examples = None
            for example in ExampleIterator(targets):
                if inferred_examples is None:
                    inferred_examples = self.learning_system.infer_examples(
                        targets, theory)
                updated = self.perform_operation_for_example(
                    example, theory, targets, inferred_examples)
                if updated:
                    inferred_examples = None
            return theory
        except KnowledgeException as e:
            raise TheoryRevisionException("Error when revising the theory.", e)

    def add_null_examples(self, targets):
        """
        Adds the null example for each predicate in `targets`.

        :param targets: the targets
        :type targets: Examples
        """
        for predicate in targets.keys():
            targets.add_example(
                get_null_example(self.learning_system.knowledge_base,
                                 predicate))

    # noinspection PyMissingOrEmptyDocstring
    def theory_revision_accepted(self, revised_theory):
        pass

    def perform_operation_for_example(self, example, theory, targets,
                                      inferred_examples):
        """
        Performs the operation for a single examples.

        :param example: the example
        :type example: Atom
        :param theory: the theory
        :type theory: NeuralLogProgram
        :param targets: the other examples
        :type targets: Examples
        :param inferred_examples: the inferred value for the examples
        :type inferred_examples: ExamplesInferences
        :return: `True`, if the theory has changed.
        :rtype: bool
        """
        try:
            if not is_positive(example) or \
                    inferred_examples.contains_example(example):
                # The inferred examples contains only the examples whose
                # weight is greater than the null weight
                if is_positive(example):
                    logger.debug("Skipping covered example:\t%s", example)
                # It skips negatives or covered positive examples
                return

            logger.debug("Building clause from the example:\t%s", example)
            bottom_clause = self.build_bottom_clause(example)
            horn_clause = \
                self.build_rule_from_bottom_clause(targets, bottom_clause)
            horn_clause = self.apply_clause_modifiers(horn_clause, targets)
            horn_clause.provenance = LearnedClause(self)
            theory.add_clauses([horn_clause])
            theory.build_program()
            logger.info("Rule appended to the theory:\t%s", horn_clause)
            return True
        except (KnowledgeException, InitializationException):
            logger.exception("Error when revising the example, reason:")
        return False

    def build_bottom_clause(self, example):
        """
        Builds a bottom clause based on the `example`.

        :param example: the example
        :type example: Atom
        :return: the bottom clause
        :rtype: HornClause
        """
        relevant_set = relevant_breadth_first_search(
            example.terms, self.relevant_depth,
            self.learning_system, not self.refine)
        variable_generator = self.variable_generator.clean_copy()
        variable_map: Dict[Term, Term] = dict()
        variable_atom = to_variable_atom(
            example, variable_generator, variable_map)
        body = []
        for atom in relevant_set:
            body.append(Literal(to_variable_atom(
                atom, variable_generator, variable_map)))

        return HornClause(variable_atom, *body)

    def build_rule_from_bottom_clause(self, targets, bottom_clause):
        """
        Builds a Horn clause from a bottom clause, based on the Guimarães and
        Paes rule creation algorithm.

        :param targets: the evaluation targets
        :type targets: Examples
        :param bottom_clause: the bottom clause
        :type bottom_clause: HornClause
        :raise KnowledgeException: in case an error occurs during the revision
        :return: a Horn clause
        :rtype: HornClause or None
        """
        logger.debug("Finding the minimal safe clauses from the bottom clause.")
        candidate_clauses = build_minimal_safe_equivalent_clauses(bottom_clause)
        logger.debug(
            "Evaluating the initial %s theory(es).", len(candidate_clauses))
        best_clause = self.multiprocessing.get_best_clause_from_candidates(
            candidate_clauses, targets)
        if best_clause is None:
            logger.debug(
                "No minimal safe clause could be evaluated. There are two "
                "possible reasons: the timeout is too low; or the metric "
                "returns the default value for all evaluations")
            return None

        if self.refine:
            best_clause = \
                self.refine_rule(best_clause, bottom_clause.body, targets)

        return best_clause.horn_clause

    def refine_rule(self, initial_clause, candidate_literals, targets):
        """
        Refines the rule.

        It starts from the `initial_clause` and adds a literal at a time,
        from `candidate_literals` into its body. At each time, getting the best
        possible Horn clause, in a greedy search.

        It finishes when one of the following criteria is met:
        1) the addition of another literal did not improve the clause in
        `self.maximum_side_way_movements` times; or
        2) there is no more possible candidates to add.

        After it finishes, it returns the best found Horn clause, based on
        the `targets`.

        :param initial_clause: the initial clause
        :type initial_clause: AsyncTheoryEvaluator[EquivalentHornClause]
        :param candidate_literals: the candidate literals
        :type candidate_literals: Collection[Literal]
        :param targets: the target examples
        :type targets: Examples
        :return: a async theory evaluator containing the best Horn clause
        :rtype: AsyncTheoryEvaluator[EquivalentHornClause]
        """
        # noinspection PyTypeChecker
        candidates = OrderedSet(candidate_literals)
        best_clause = initial_clause
        current_clause = initial_clause
        remove_equivalent_candidates(candidates, initial_clause.element)
        side_way_movements = 0
        logger.debug("Refining rule:\t%s", initial_clause.horn_clause)
        while not self.is_to_stop_by_side_way_movements(side_way_movements) \
                and candidates:
            remove_last_literal_equivalent_candidates(
                candidates, current_clause.element)
            current_clause = self.specify_rule(
                current_clause.element, candidates, targets)
            if current_clause is None:
                break
            improvement = self.theory_metric.difference(
                current_clause.evaluation, best_clause.evaluation)
            if improvement > self.improvement_threshold:
                # Current clause is better than the best clause, making it
                # the best clause
                logger.debug("Accepting new best refined candidate:\t%s",
                             current_clause.horn_clause)
                best_clause = current_clause
                side_way_movements = 0
            else:
                # Current clause is not better than the best clause
                logger.debug("Making side movement for candidate:\t%s",
                             current_clause.element)
                side_way_movements += 1
                if improvement >= 0.0 and not self.generic:
                    # There current close is not worst than the best clause,
                    # and is more specific than the best one. Since the generic
                    # flag is `False`, make the current clause the best one
                    best_clause = current_clause

        return best_clause

    def is_to_stop_by_side_way_movements(self, side_way_movements):
        """
        Checks if it is to stop due to reaching the maximum side way movements.

        :param side_way_movements: the number of iterations without improvement
        :type side_way_movements: int
        :return: `True`, if it is to stop due to the maximum side way
        movements; otherwise, `False`
        :rtype: bool
        """
        return 0 <= self.maximum_side_way_movements < side_way_movements

    def specify_rule(self, clause, candidates, targets):
        """
        Makes the Horn clause more specific by adding a literal from
        `candidates` into its body. All the possible literals are tested,
        and the best one, based on the `targets` is returned.

        :param clause: the clause
        :type clause: EquivalentHornClause
        :param candidates: the candidates
        :type candidates: Set[Literal]
        :param targets: the target examples
        :type targets: Examples
        :return: the best obtained clause
        :rtype: AsyncTheoryEvaluator[EquivalentHornClause]
        """
        return self.multiprocessing.get_best_clause_from_candidates(
            clause.build_appended_candidates(candidates), targets)


class CombinedBottomClauseBoundedRule(BottomClauseBoundedRule):
    """
    Class to create a single rule from a set of examples by choosing literals
    from the combined bottom clause from all the examples.
    """

    # noinspection PyMissingOrEmptyDocstring
    def perform_operation(self, targets):
        try:
            logger.info("Performing operation on\t%d examples.", len(targets))
            theory = self.learning_system.theory.copy()
            self.perform_operation_for_examples(targets, theory)
            return theory
        except TheoryRevisionException:
            logger.exception("Error when copying the theory.")

    def perform_operation_for_examples(self, targets, theory):
        """
        Performs the operation for all the examples combined.

        :param targets: the examples
        :type targets: Examples
        :param theory: the theory
        :type theory: NeuralLogProgram
        """
        logger.info("Number of predicates found among the examples:\t%d",
                    len(targets))
        for predicate, examples in targets.items():
            try:
                logger.info("Building rule for predicate\t%s and\t%d examples.",
                            predicate, len(examples))
                examples = Examples({predicate: examples})
                bottom_clause = self.build_combined_bottom_clause(
                    predicate, examples)
                logger.info("Bottom clause body size:\t%d",
                            len(bottom_clause.body))
                new_rule = self.build_rule_from_bottom_clause(
                    targets, bottom_clause)
                new_rule = self.apply_clause_modifiers(new_rule, examples)
                theory.add_clauses([new_rule])
                theory.build_program()
                logger.info("Rule appended to the theory:\t%s", new_rule)
            except TheoryRevisionException as e:
                logger.debug("Error when revising the theory, reason:\t%s", e)

    def build_combined_bottom_clause(self, predicate, examples):
        """
        Builds the bottom clause from the combination of the bottom clause
        from the examples.

        :param predicate: the predicate of the examples
        :type predicate: Predicate
        :param examples: the examples
        :type examples: Examples
        :return: the combined bottom clause
        :rtype: HornClause
        """
        examples = examples.get(predicate)
        positive_terms = map(lambda x: x.terms, examples.values())
        positive_terms = set(itertools.chain.from_iterable(positive_terms))
        relevant_set = relevant_breadth_first_search(
            positive_terms, self.relevant_depth, self.learning_system,
            not self.refine)
        positive_examples = set(filter(is_positive, examples.values()))

        return self.build_variable_bottom_clause(
            predicate, relevant_set, positive_examples)

    def build_variable_bottom_clause(
            self, predicate, relevant_set, positive_examples):
        """
        Builds the variable bottom clause from the relevant set and the grounded
        examples.

        :param predicate: the predicate
        :type predicate: Predicate
        :param relevant_set: the relevant set of atoms
        :type relevant_set: Set[Atom]
        :param positive_examples: the positive examples
        :type positive_examples: Set[Atom]
        :return: the variable bottom clause combined from the examples
        :rtype: HornClause
        """
        variable_map: Dict[Term, Term] = dict()
        variable_generator = self.variable_generator.clean_copy()
        body = []
        example_variables = list(map(lambda x: next(variable_generator),
                                     range(predicate.arity)))
        for atom in positive_examples:
            add_variable_substitutions(atom, variable_map, example_variables)

        for atom in relevant_set:
            body.append(
                Literal(
                    to_variable_atom(atom, variable_generator, variable_map)))

        return HornClause(Atom(predicate, *example_variables), *body)


class CombinedBottomClauseBreadthSearch(CombinedBottomClauseBoundedRule):
    """
    Class to create a set of rules from a set of examples by doing a breadth
    search on the space of the combined bottom clause from all the examples.

    It appends a literal at a time until creates all the possible clauses of
    size `self.maximum_size`. Then, it evaluates those clauses and split
    them in the biggest difference between the evaluation of the `n` first
    clauses and the `n + 1` last clauses; returning the first `n` clauses.
    """

    OPTIONAL_FIELDS = CombinedBottomClauseBoundedRule.OPTIONAL_FIELDS
    OPTIONAL_FIELDS.update({
        "maximum_size": 2,
        "strict_to_maximum_size": False
    })

    def __init__(self,
                 learning_system=None,
                 theory_metric=None,
                 variable_generator=None,
                 clause_modifiers=None,
                 relevant_depth=0,
                 refine=False,
                 maximum_side_way_movements=-1,
                 improvement_threshold=0.0,
                 generic=True,
                 evaluation_timeout=300,
                 number_of_process=1,
                 maximum_size=2,
                 strict_to_maximum_size=False):
        """
        Creates a Combined Bottom Clause Breadth Search operator.

        :param learning_system: the learning system
        :type learning_system: sls.StructureLearningSystem
        :param theory_metric: the theory metric
        :type theory_metric: TheoryMetric
        :param clause_modifiers: a clause modifier, a list of clause modifiers
        or none
        :type clause_modifiers: ClauseModifier or Collection[ClauseModifier]
        or None
        :param variable_generator: the variable generator
        :type variable_generator: Optional[VariableGenerator]
        :param relevant_depth: the relevant depth
        :type relevant_depth: Optional[int]
        :param refine: if it is to refine the rules
        :type refine: Optional[bool]
        :param maximum_side_way_movements: the maximum side way movements
        :type maximum_side_way_movements: Optional[int]
        :param improvement_threshold: the improvement threshold
        :type improvement_threshold: Optional[float]
        :param generic: if it is to return the most generic rule
        :type generic: Optional[bool]
        :param evaluation_timeout: the evaluation timeout, in seconds
        :type evaluation_timeout: Optional[int]
        :param number_of_process: the number of parallel process
        :type number_of_process: Optional[int]
        :param maximum_size: the maximum size of the clauses.
        :type maximum_size: Optional[int]
        :param strict_to_maximum_size: If `True`, it creates only rules of the
        maximum size. Otherwise, it looks for improvements until one of the
        three criteria is met: (1) the maximum size is reached; (2) the
        theory does not improve over `maximum_side_way_movements` iterations;
        (3) there are no more candidates to test
        :type strict_to_maximum_size: Optional[bool]
        """
        super().__init__(learning_system, theory_metric, clause_modifiers,
                         variable_generator, relevant_depth, refine,
                         maximum_side_way_movements, improvement_threshold,
                         generic, evaluation_timeout, number_of_process)
        self.maximum_size = maximum_size
        if maximum_size is None:
            self.maximum_size = self.OPTIONAL_FIELDS["maximum_size"]

        self.strict_to_maximum_size = strict_to_maximum_size
        if strict_to_maximum_size is None:
            self.strict_to_maximum_size = \
                self.OPTIONAL_FIELDS["strict_to_maximum_size"]

    # noinspection PyMissingOrEmptyDocstring
    def perform_operation_for_examples(self, targets, theory):
        logger.info("Number of predicates found among the examples:\t%d",
                    len(targets))
        for predicate, examples in targets.items():
            try:
                logger.info("Building rule for predicate\t%s and\t%s examples.",
                            predicate, len(examples))
                examples = Examples({predicate: examples})
                bottom_clause = self.build_combined_bottom_clause(
                    predicate, examples)
                logger.info("Bottom clause body size:\t%d",
                            len(bottom_clause.body))
                self.build_rules_from_bottom_clause(
                    targets, bottom_clause, theory)
            except TheoryRevisionException as e:
                logger.debug("Error when revising the theory, reason:\t%s", e)

    def build_rules_from_bottom_clause(self, targets, bottom_clause, theory):
        """
        Builds a Horn clause from a bottom clause, based on the Guimarães and
        Paes rule creation algorithm.

        :param targets: the target examples
        :type targets: Examples
        :param bottom_clause: the bottom clause
        :type bottom_clause: HornClause
        :param theory: the theory
        :type theory: NeuralLogProgram
        """
        candidate_literals = filter(lambda x: not x.negated, bottom_clause.body)
        candidate_literals = sorted(
            candidate_literals,
            key=lambda x: (x.predicate.name, x.predicate.arity))
        candidate_literals = list(candidate_literals)
        horn_clauses = self.refine_rules(
            candidate_literals, targets, bottom_clause)
        for clause in horn_clauses:
            modified_clause = self.apply_clause_modifiers(clause, targets)
            theory.add_clauses([modified_clause])
            logger.info("Rule appended to the theory:\t%s", modified_clause)
        theory.build_program()

    def refine_rules(self, candidates, targets, bottom_clause):
        """
        Refines the set of clauses

        :param candidates: the candidate literals
        :type candidates: List[Literal]
        :param targets: the evaluation examples
        :type targets: Examples
        :param bottom_clause: the initial bottom clause
        :type bottom_clause: HornClause
        :return: the best set of clauses
        :rtype: List[HornClause]
        """
        queue: Deque[EquivalentHornClause] = deque()
        queue.append(EquivalentHornClause(bottom_clause.head))
        size = self._get_size(bottom_clause)
        i = 1
        side_way_movements = 0
        best_evaluation = self.theory_metric.default_value
        best_clauses = []

        while not self.is_to_stop_by_side_way_movements(side_way_movements) \
                and queue and i < size:
            current_clauses = self.get_next_clauses(
                candidates, targets, queue, i)
            if not current_clauses:
                break
            candidate_string = iterable_to_string(current_clauses)
            logger.info("Candidate clauses:\n%s", candidate_string)
            current_evaluation = self.evaluate_clauses(current_clauses, targets)
            logger.debug("Candidates evaluation:\t%f", current_evaluation)
            improvement = self.theory_metric.difference(
                current_evaluation, best_evaluation)
            if improvement > self.improvement_threshold:
                logger.debug("Accepting new best refined candidate:\n%s",
                             candidate_string)
                best_evaluation = current_evaluation
                best_clauses = current_clauses
                side_way_movements = 0
            else:
                logger.debug("Making side movement for candidate:\n%s",
                             candidate_string)
                side_way_movements += 1
                if improvement >= 0.0 and not self.generic:
                    best_clauses = current_clauses
            if self.strict_to_maximum_size:
                break
            i += 1

        return best_clauses

    def _get_size(self, bottom_clause):
        return len(
            bottom_clause.body) if self.maximum_size < 1 else self.maximum_size

    def get_next_clauses(self, candidates, targets, queue, index):
        """
        Gets the next iteration of clauses.

        :param candidates: the candidate literals
        :type candidates: List[Literal]
        :param targets: the evaluation examples
        :type targets: Examples
        :param queue: the queue of clauses
        :type queue: deque[EquivalentHornClause]
        :param index: the index of the iteration
        :type index: int
        :return: the next iteration of clauses
        :rtype: List[HornClause]
        """
        if self.strict_to_maximum_size:
            for j in range(index, self.maximum_size + 1):
                logger.debug("Finding the clauses, from the bottom clause, "
                             "of size:\t%d", j)
                append_all_candidates_to_queue(queue, candidates)
        else:
            logger.debug("Finding the clauses, from the bottom clause, "
                         "of size:\t%d", index)
            append_all_candidates_to_queue(queue, candidates)

        if not queue:
            return []

        logger.debug("Evaluating %s theory(es) of size:\t%d", len(queue), index)
        evaluation_map: \
            Dict[AsyncTheoryEvaluator[EquivalentHornClause], float] = dict()
        self.multiprocessing.get_best_clause_from_candidates(
            queue, targets, evaluation_map)
        equivalent_horn_clauses = \
            sorted(
                evaluation_map.items(),
                key=my_cmp_to_key(
                    self.theory_metric.compare, key=lambda x: x[1]))
        equivalent_horn_clauses = \
            list(map(lambda x: x[0], equivalent_horn_clauses))
        biggest_gap = find_biggest_gap(
            equivalent_horn_clauses, lambda x: x.evaluation)
        logger.debug("The biggest gap threshold was:\t%f",
                     equivalent_horn_clauses[biggest_gap - 1].evaluation)
        equivalent_horn_clauses = equivalent_horn_clauses[:biggest_gap]

        return list(map(lambda x: x.horn_clause, equivalent_horn_clauses))

    def evaluate_clauses(self, clauses, targets):
        """
        Evaluates the clauses on the targets.

        :param clauses: the clauses
        :type clauses: Collection[HornClause]
        :param targets: the targets
        :type targets: Examples
        :return: the evaluation of the clauses on the targets
        :rtype: float
        """
        theory_evaluator = self.learning_system.theory_evaluator
        return theory_evaluator.evaluate_theory_appending_clause(
            targets, self.theory_metric, clauses)
