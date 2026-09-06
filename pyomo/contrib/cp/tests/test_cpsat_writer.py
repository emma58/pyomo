# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.common.unittest as unittest

from pyomo.contrib.cp import IntervalVar, SequenceVar, Pulse, Step, AlwaysIn
from pyomo.contrib.cp.scheduling_expr.scheduling_logic import (
    spans,
    alternative,
    synchronize,
)
from pyomo.contrib.cp.scheduling_expr.sequence_expressions import (
    predecessor_to,
    before_in_sequence,
    no_overlap,
)
from pyomo.contrib.cp.repn.cpsat_writer import cp_model_available
from pyomo.contrib.cp.tests import common_tests as ct
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
    implies,
    land,
    exactly,
    atleast,
    atmost,
    all_different,
    count_if,
)
from pyomo.environ import (
    ConcreteModel,
    Set,
    Var,
    Integers,
    BooleanVar,
    LogicalConstraint,
    Constraint,
    Objective,
    maximize,
    value,
    TerminationCondition,
)
from pyomo.opt import SolverFactory


@unittest.skipIf(not cp_model_available, "ortools is not available")
class TestSolveModel(unittest.TestCase):
    def test_solve_infeasible_problem(self):
        m = ConcreteModel()
        m.x = Var(within=[1, 2, 3, 5])
        m.c = Constraint(expr=m.x == 0)

        result = SolverFactory('cp_sat').solve(m)
        self.assertEqual(
            result.solver.termination_condition, TerminationCondition.infeasible
        )

    def test_solve_max_problem(self):
        m = ConcreteModel()
        m.cookies = Var(domain=Integers, bounds=(7, 10))
        m.chocolate_chip_equity = Constraint(expr=m.cookies <= 9)
        m.obj = Objective(expr=m.cookies, sense=maximize)

        results = SolverFactory('cp_sat').solve(m)

        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(value(m.cookies), 9)
        self.assertEqual(results.problem.lower_bound, 9)
        self.assertEqual(results.problem.upper_bound, 9)

    def test_algebraic_operators(self):
        # product, abs, min, max, ranged, truncating division -- all built
        # via an auxiliary target var + the matching add_*_equality call,
        # since CP-SAT's expression objects don't overload these directly
        # the way docplex's do.
        m = ConcreteModel()
        m.x = Var(bounds=(2, 5), domain=Integers)
        m.y = Var(bounds=(3, 4), domain=Integers)
        m.p = Var(bounds=(0, 20), domain=Integers)
        m.x.fix(3)
        m.y.fix(4)
        m.prod = Constraint(expr=m.p == m.x * m.y)

        m.a = Var([1, 2, 3], bounds=(0, 10), domain=Integers)
        m.a[1].fix(3)
        m.a[2].fix(7)
        m.a[3].fix(1)
        m.mn = Var(bounds=(0, 10), domain=Integers)
        m.mx = Var(bounds=(0, 10), domain=Integers)
        m.c1 = Constraint(expr=m.mn == MinExpression([m.a[1], m.a[2], m.a[3]]))
        m.c2 = Constraint(expr=m.mx == MaxExpression([m.a[1], m.a[2], m.a[3]]))

        m.q = Var(bounds=(0, 20), domain=Integers)
        m.qcon = Constraint(expr=m.q == m.p / m.y)  # 12 // 4 == 3

        m.r = Var(bounds=(0, 20), domain=Integers)
        m.rcon = Constraint(expr=(5, m.r, 15))

        results = SolverFactory('cp_sat').solve(m)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(value(m.p), 12)
        self.assertEqual(value(m.mn), 1)
        self.assertEqual(value(m.mx), 7)
        self.assertEqual(value(m.q), 3)
        self.assertTrue(5 <= value(m.r) <= 15)

    def test_all_different(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3], bounds=(0, 2), domain=Integers)
        m.con = LogicalConstraint(expr=all_different(m.x[i] for i in [1, 2, 3]))

        results = SolverFactory('cp_sat').solve(m)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(sorted(value(m.x[i]) for i in [1, 2, 3]), [0, 1, 2])

    def test_exactly_atleast_atmost(self):
        m = ConcreteModel()
        m.b = BooleanVar([1, 2, 3, 4])
        m.con = LogicalConstraint(expr=exactly(2, m.b[1], m.b[2], m.b[3], m.b[4]))
        SolverFactory('cp_sat').solve(m)
        self.assertEqual(sum(1 for i in [1, 2, 3, 4] if value(m.b[i])), 2)

        m2 = ConcreteModel()
        m2.b = BooleanVar([1, 2, 3, 4])
        m2.con = LogicalConstraint(expr=atleast(3, m2.b[1], m2.b[2], m2.b[3], m2.b[4]))
        SolverFactory('cp_sat').solve(m2)
        self.assertGreaterEqual(sum(1 for i in [1, 2, 3, 4] if value(m2.b[i])), 3)

        m3 = ConcreteModel()
        m3.b = BooleanVar([1, 2, 3, 4])
        for i in [1, 2, 3]:
            m3.b[i].fix(True)
        m3.con = LogicalConstraint(expr=atmost(1, m3.b[1], m3.b[2], m3.b[3], m3.b[4]))
        results = SolverFactory('cp_sat').solve(m3)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.infeasible
        )

    def test_count_if(self):
        m = ConcreteModel()
        m.b = BooleanVar([1, 2, 3])
        m.b[1].fix(True)
        m.b[2].fix(True)
        m.b[3].fix(False)
        m.cnt = Var(domain=Integers, bounds=(0, 3))
        m.con = LogicalConstraint(expr=m.cnt == count_if(m.b[i] for i in [1, 2, 3]))

        SolverFactory('cp_sat').solve(m)
        self.assertEqual(value(m.cnt), 2)

    def test_nested_logical_constraint(self):
        # Exercises the _AUXILIARY reify/materialize path (a fresh literal
        # gets minted and tied to the nested subexpression's truth value),
        # not just the cheaper root-level assert path.
        m = ConcreteModel()
        m.a = BooleanVar()
        m.b = BooleanVar()
        m.c = BooleanVar()
        m.a.fix(True)
        m.con = LogicalConstraint(expr=implies(m.a, land(m.b, m.c)))

        SolverFactory('cp_sat').solve(m)
        self.assertTrue(value(m.b))
        self.assertTrue(value(m.c))

    def test_get_item_expression_indirection(self):
        m = ConcreteModel()
        m.i = IntervalVar(
            [1, 2, 3], optional=True, start=(0, 10), end=(0, 10), length=(0, 10)
        )
        m.x = Var(within={1, 2, 3})
        m.cons = LogicalConstraint(expr=m.i[m.x].is_present)

        results = SolverFactory('cp_sat').solve(m)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertTrue(value(m.i[int(value(m.x))].is_present))

    def test_span_expression(self):
        m = ConcreteModel()
        m.a = IntervalVar(start=(0, 10), end=(0, 10), length=(1, 10))
        m.b = IntervalVar(start=(0, 10), end=(0, 10), length=2)
        m.c = IntervalVar(start=(0, 10), end=(0, 10), length=3)
        m.b.start_time.fix(2)
        m.c.start_time.fix(5)
        m.con = LogicalConstraint(expr=spans(m.a, m.b, m.c))

        SolverFactory('cp_sat').solve(m)
        self.assertEqual(value(m.a.start_time), 2)
        self.assertEqual(value(m.a.end_time), 8)

    def test_alternative_expression(self):
        m = ConcreteModel()
        m.container = IntervalVar(start=(0, 10), end=(0, 10), length=(1, 10))
        m.opt1 = IntervalVar(optional=True, start=(0, 10), end=(0, 10), length=3)
        m.opt2 = IntervalVar(optional=True, start=(0, 10), end=(0, 10), length=5)
        m.opt1.start_time.fix(2)
        m.opt2.start_time.fix(4)
        m.con = LogicalConstraint(expr=alternative(m.container, m.opt1, m.opt2))

        SolverFactory('cp_sat').solve(m)
        self.assertTrue(value(m.container.is_present))
        self.assertEqual(value(m.opt1.is_present) + value(m.opt2.is_present), 1)
        if value(m.opt1.is_present):
            self.assertEqual(value(m.container.start_time), 2)
            self.assertEqual(value(m.container.end_time), 5)
        else:
            self.assertEqual(value(m.container.start_time), 4)
            self.assertEqual(value(m.container.end_time), 9)

    def test_synchronize_expression(self):
        m = ConcreteModel()
        m.container = IntervalVar(start=(0, 10), end=(0, 10), length=5)
        m.container.start_time.fix(3)
        m.follower = IntervalVar(
            optional=True, start=(0, 10), end=(0, 10), length=(1, 10)
        )
        m.con = LogicalConstraint(expr=synchronize(m.container, m.follower))

        SolverFactory('cp_sat').solve(m)
        self.assertTrue(value(m.follower.is_present))
        self.assertEqual(value(m.follower.start_time), 3)
        self.assertEqual(value(m.follower.end_time), 8)

    def test_scheduling_with_sequence_vars(self):
        # Exercises Branch 1 (start-time comparisons): first_in_sequence and
        # two predecessor_to constraints, with an accompanying no_overlap --
        # the same model docplex's writer is tested against.
        ct.check_solve_three_step_sequence_model(self, 'cp_sat')

    def test_sequencing_without_no_overlap(self):
        # No docplex-suite analog: exercises Branch 2 (explicit rank/
        # position variables), which is only needed when no NoOverlap over
        # the same SequenceVar guarantees a temporal ordering to piggyback
        # on. All three intervals are free to overlap in time here; only
        # their *sequence position* is constrained.
        from pyomo.contrib.cp.tests.models import (
            three_step_sequence_model_no_overlap_constraint,
        )

        m = three_step_sequence_model_no_overlap_constraint()
        results = SolverFactory('cp_sat').solve(m)
        self.assertIn(
            results.solver.termination_condition,
            (TerminationCondition.optimal, TerminationCondition.feasible),
        )

    def test_predecessor_to_forbids_interloper(self):
        # The trickiest part of Branch 1's encoding: predecessor_to means
        # *direct* adjacency, so nothing else present may be scheduled
        # strictly between the two intervals -- unlike before_in_sequence,
        # which only requires "somewhere earlier," not "immediately before."
        def build(use_predecessor):
            m = ConcreteModel()
            m.a = IntervalVar(start=(0, 0), end=(2, 2), length=2)
            m.c = IntervalVar(start=(5, 5), end=(7, 7), length=2)
            # b's bounds force it into the gap between a and c.
            m.b = IntervalVar(start=(3, 3), end=(4, 4), length=1)
            m.seq = SequenceVar(expr=[m.a, m.b, m.c])
            if use_predecessor:
                m.pred = LogicalConstraint(expr=predecessor_to(m.a, m.c, m.seq))
            else:
                m.pred = LogicalConstraint(expr=before_in_sequence(m.a, m.c, m.seq))
            m.no_ovl = LogicalConstraint(expr=no_overlap(m.seq))
            return m

        m1 = build(use_predecessor=True)
        results1 = SolverFactory('cp_sat').solve(m1)
        self.assertEqual(
            results1.solver.termination_condition, TerminationCondition.infeasible
        )

        m2 = build(use_predecessor=False)
        results2 = SolverFactory('cp_sat').solve(m2)
        self.assertIn(
            results2.solver.termination_condition,
            (TerminationCondition.optimal, TerminationCondition.feasible),
        )

    def test_pulse_cumulative_fast_path(self):
        m = ConcreteModel()
        m.tasks = Set(initialize=[0, 1, 2])
        m.t = IntervalVar(m.tasks, start=(0, 10), end=(0, 10), length=3)
        m.usage = sum(Pulse((m.t[i], 1)) for i in m.tasks)
        m.cap = LogicalConstraint(
            expr=AlwaysIn(cumul_func=m.usage, bounds=(0, 2), times=(0, 10))
        )

        results = SolverFactory('cp_sat').solve(m)
        self.assertIn(
            results.solver.termination_condition,
            (TerminationCondition.optimal, TerminationCondition.feasible),
        )
        starts = [value(m.t[i].start_time) for i in m.tasks]
        for t in range(10):
            n_active = sum(1 for i in m.tasks if starts[i] <= t < starts[i] + 3)
            self.assertLessEqual(n_active, 2)

    def test_step_only_reservoir_special_case(self):
        m = ConcreteModel()
        m.produce = IntervalVar(start=(0, 10), end=(0, 10), length=1)
        m.consume = IntervalVar(start=(0, 10), end=(0, 10), length=1)
        m.level = Step(m.produce.start_time, 5) - Step(m.consume.start_time, 5)
        m.cap = LogicalConstraint(
            expr=AlwaysIn(cumul_func=m.level, bounds=(0, 5), times=(0, 10))
        )
        m.order = LogicalConstraint(
            expr=m.produce.start_time.before(m.consume.start_time)
        )

        results = SolverFactory('cp_sat').solve(m)
        self.assertIn(
            results.solver.termination_condition,
            (TerminationCondition.optimal, TerminationCondition.feasible),
        )
        self.assertLessEqual(value(m.produce.start_time), value(m.consume.start_time))

    def test_always_in_unsupported_general_case_raises_not_implemented(self):
        # A CumulativeFunction mixing Pulse and Step terms is outside the two
        # special cases this writer supports (see the "Scheduling: step
        # functions" section of cpsat_writer.py) -- confirmed to raise a
        # clear, actionable error rather than silently mistranslating it.
        m = ConcreteModel()
        m.a = IntervalVar(start=(0, 10), end=(0, 10), length=2)
        m.mixed = Pulse((m.a, 1)) + Step(m.a.start_time, 1)
        m.con = LogicalConstraint(
            expr=AlwaysIn(cumul_func=m.mixed, bounds=(0, 5), times=(0, 10))
        )

        with self.assertRaises(NotImplementedError):
            SolverFactory('cp_sat').solve(m)

    def test_unbounded_interval_var_raises_clear_error(self):
        # Unlike CP Optimizer, CP-SAT has no notion of an unbounded horizon:
        # every IntervalVar's start/end/length needs finite bounds.
        m = ConcreteModel()
        m.i = IntervalVar(length=1, end=(0, 24))  # no start= given -> unbounded

        with self.assertRaises(ValueError):
            SolverFactory('cp_sat').solve(m)
