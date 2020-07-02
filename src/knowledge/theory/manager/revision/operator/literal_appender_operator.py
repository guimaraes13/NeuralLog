"""
Handles some revision operators that append literals to the clause.
"""
import collections
from abc import abstractmethod
from random import random
from typing import TypeVar, Generic, Set, Dict

from src.knowledge.examples import Examples, ExampleIterator, ExamplesInferences
from src.knowledge.theory.manager.revision.operator.revision_operator import \
    RevisionOperator, is_positive, relevant_breadth_first_search
from src.language.equivalent_clauses import EquivalentAtom
from src.language.language import HornClause, get_variable_atom, Literal, \
    Atom, \
    Term
from src.util.clause_utils import to_variable_atom
from src.util.multiprocessing.multiprocessing import \
    DEFAULT_NUMBER_OF_PROCESS, \
    DEFAULT_EVALUATION_TIMEOUT
from src.util.multiprocessing.theory_evaluation import AsyncTheoryEvaluator, \
    SyncTheoryEvaluator
from src.util.variable_generator import VariableGenerator

V = TypeVar('V')


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
            self, initial_clause, substitution_goal, examples, inferences,
            skip_candidates, connected):
        """
        Gets the literal candidates from the examples. The literals that are
        candidates to be appended to the initial clause in order to get
        improve it.

        :param initial_clause: the initial clause
        :type initial_clause: HornClause
        :param substitution_goal: the substitution query
        :type substitution_goal: Atom
        :param examples: the examples inferred by the substitution query
        :type examples: Examples
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
                example = examples.get(predicate, dict()).get(key)
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
