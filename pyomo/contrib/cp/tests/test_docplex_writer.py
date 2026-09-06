# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.common.fileutils import Executable

from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.repn.docplex_writer import LogicalToDoCplex
from pyomo.contrib.cp.tests import common_tests as ct
from pyomo.environ import (
    all_different,
    count_if,
    ConcreteModel,
    Set,
    Var,
    Integers,
    Param,
    LogicalConstraint,
    value,
    TerminationCondition,
    Constraint,
    PositiveIntegers,
    maximize,
    minimize,
    Objective,
)
from pyomo.opt import WriterFactory, SolverFactory

try:
    import docplex.cp.model as cp

    docplex_available = True
except:
    docplex_available = False

cpoptimizer_available = Executable('cpoptimizer').available()


@unittest.skipIf(not docplex_available, "docplex is not available")
class TestWriteModel(unittest.TestCase):
    def test_write_scheduling_model_only_interval_vars(self):
        m = ConcreteModel()
        m.i = IntervalVar(start=(2, 4), end=(5, 19), length=7, optional=False)
        m.tasks = Set(initialize=range(2))
        m.h = IntervalVar(m.tasks, optional=True, length=(4, 5), start=(1, 2))

        cpx_mod, var_map = WriterFactory('docplex_model').write(m)

        # We have nothing on this model other than interval vars
        exprs = cpx_mod.get_all_expressions()
        self.assertEqual(len(exprs), 3)

        # We should have the three interval vars above
        variables = cpx_mod.get_all_variables()
        self.assertEqual(len(variables), 3)
        # I'm assuming that the lists of exprs and vars are in a deterministic
        # order. If they're not, this will fail periodically, so I guess we'll
        # find out.
        self.assertIs(variables[0], var_map[m.h[1]])
        self.assertIs(exprs[2][0], var_map[m.h[1]])
        self.assertIs(variables[1], var_map[m.h[0]])
        self.assertIs(exprs[1][0], var_map[m.h[0]])

        for i in [0, 1]:
            self.assertTrue(variables[i].is_optional())
            self.assertEqual(variables[i].get_start(), (1, 2))
            self.assertEqual(variables[i].get_length(), (4, 5))

        self.assertIs(variables[2], var_map[m.i])
        self.assertIs(exprs[0][0], var_map[m.i])
        self.assertTrue(variables[2].is_present())
        self.assertEqual(variables[2].get_start(), (2, 4))
        self.assertEqual(variables[2].get_end(), (5, 19))
        self.assertEqual(variables[2].get_length(), (7, 7))

    def test_write_model_with_bool_expr_as_constraint(self):
        # This tests our handling of a quirk with docplex that even some things
        # that are boolean-valued can't be added to the model as constraints. We
        # need to explicitly recognize them and add an "== True" right-hand
        # side.
        m = ConcreteModel()
        m.i = IntervalVar([1, 2], optional=True)
        m.x = Var(within={1, 2})
        # This is a perfectly reasonable constraint in a context where x is some
        # variable that decides what needs to be scheduled.
        m.cons = LogicalConstraint(expr=m.i[m.x].is_present)

        cpx_mod, var_map = WriterFactory('docplex_model').write(m)

        variables = cpx_mod.get_all_variables()
        self.assertEqual(len(variables), 3)
        # The three variables plus the one constraint:
        exprs = cpx_mod.get_all_expressions()
        self.assertEqual(len(exprs), 4)

        x = var_map[m.x]
        i1 = var_map[m.i[1]]
        i2 = var_map[m.i[2]]

        self.assertIs(variables[0], x)
        self.assertIs(variables[1], i2)
        self.assertIs(variables[2], i1)

        self.assertTrue(
            exprs[3][0].equals(
                cp.element(
                    [cp.presence_of(i1), cp.presence_of(i2)], 0 + 1 * (x - 1) // 1
                )
                == True
            )
        )


@unittest.skipIf(not docplex_available, "docplex is not available")
@unittest.skipIf(not cpoptimizer_available, "CP optimizer is not available")
class TestSolveModel(unittest.TestCase):
    def test_solve_scheduling_problem(self):
        results = ct.check_solve_mice_and_cookies_model(self, 'cp_optimizer')

        # docplex-specific problem-size stats (these aren't something we'd
        # expect to match across backends, since different writers encode
        # the same model with different numbers of native variables/
        # constraints)
        self.assertEqual(results.problem.number_of_constraints, 5)
        self.assertEqual(results.problem.number_of_integer_vars, 1)
        self.assertEqual(results.problem.number_of_interval_vars, 5)

    def test_solve_infeasible_problem(self):
        m = ConcreteModel()
        m.x = Var(within=[1, 2, 3, 5])
        m.c = Constraint(expr=m.x == 0)

        result = SolverFactory('cp_optimizer').solve(m)
        self.assertEqual(
            result.solver.termination_condition, TerminationCondition.infeasible
        )

        self.assertIsNone(m.x.value)

    def test_solve_max_problem(self):
        m = ConcreteModel()
        m.cookies = Var(domain=PositiveIntegers, bounds=(7, 10))
        m.chocolate_chip_equity = Constraint(expr=m.cookies <= 9)

        m.obj = Objective(expr=m.cookies, sense=maximize)

        results = SolverFactory('cp_optimizer').solve(m)

        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(value(m.cookies), 9)

        self.assertEqual(results.problem.number_of_objectives, 1)
        self.assertEqual(results.problem.sense, maximize)
        self.assertEqual(results.problem.lower_bound, 9)
        self.assertEqual(results.problem.upper_bound, 9)

    def test_solve_min_problem(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3], bounds=(4, 6), domain=Integers)
        m.y = Var(within=[1, 2, 3])

        m.c1 = Constraint(expr=m.y >= 2.5)

        @m.Constraint([1, 2, 3])
        def x_bounds(m, i):
            return m.x[i] >= 3 * (i - 1)

        m.obj = Objective(expr=m.x[m.y])

        results = SolverFactory('cp_optimizer').solve(m)

        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(value(m.x[3]), 6)
        self.assertEqual(value(m.y), 3)

        self.assertEqual(results.problem.number_of_objectives, 1)
        self.assertEqual(results.problem.sense, minimize)
        self.assertEqual(results.problem.lower_bound, 6)
        self.assertEqual(results.problem.upper_bound, 6)

    def test_matching_problem(self):
        m = ConcreteModel()

        m.People = Set(initialize=['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'])
        m.Languages = Set(initialize=['English', 'Spanish', 'Hindi', 'Swedish'])
        # People have integer names because we don't have categorical vars yet.
        m.Names = Set(initialize=range(len(m.People)))

        m.Observed = Param(
            m.Names,
            m.Names,
            m.Languages,
            initialize={
                (0, 1, 'English'): 1,
                (1, 0, 'English'): 1,
                (0, 2, 'English'): 1,
                (2, 0, 'English'): 1,
                (0, 3, 'English'): 1,
                (3, 0, 'English'): 1,
                (0, 4, 'English'): 1,
                (4, 0, 'English'): 1,
                (0, 5, 'English'): 1,
                (5, 0, 'English'): 1,
                (0, 6, 'English'): 1,
                (6, 0, 'English'): 1,
                (1, 2, 'Spanish'): 1,
                (2, 1, 'Spanish'): 1,
                (1, 5, 'Hindi'): 1,
                (5, 1, 'Hindi'): 1,
                (1, 6, 'Hindi'): 1,
                (6, 1, 'Hindi'): 1,
                (2, 3, 'Swedish'): 1,
                (3, 2, 'Swedish'): 1,
                (3, 4, 'English'): 1,
                (4, 3, 'English'): 1,
            },
            default=0,
            mutable=True,
        )  # TODO: shouldn't need to
        # be mutable, but waiting
        # on #3045

        m.Expected = Param(
            m.People,
            m.People,
            m.Languages,
            initialize={
                ('P1', 'P2', 'English'): 1,
                ('P2', 'P1', 'English'): 1,
                ('P1', 'P3', 'English'): 1,
                ('P3', 'P1', 'English'): 1,
                ('P1', 'P4', 'English'): 1,
                ('P4', 'P1', 'English'): 1,
                ('P1', 'P5', 'English'): 1,
                ('P5', 'P1', 'English'): 1,
                ('P1', 'P6', 'English'): 1,
                ('P6', 'P1', 'English'): 1,
                ('P1', 'P7', 'English'): 1,
                ('P7', 'P1', 'English'): 1,
                ('P2', 'P3', 'Spanish'): 1,
                ('P3', 'P2', 'Spanish'): 1,
                ('P2', 'P6', 'Hindi'): 1,
                ('P6', 'P2', 'Hindi'): 1,
                ('P2', 'P7', 'Hindi'): 1,
                ('P7', 'P2', 'Hindi'): 1,
                ('P3', 'P4', 'Swedish'): 1,
                ('P4', 'P3', 'Swedish'): 1,
                ('P4', 'P5', 'English'): 1,
                ('P5', 'P4', 'English'): 1,
            },
            default=0,
            mutable=True,
        )  # TODO: shouldn't need to be mutable, but
        # waiting on #3045

        m.person_name = Var(m.People, bounds=(0, max(m.Names)), domain=Integers)

        m.one_to_one = LogicalConstraint(
            expr=all_different(m.person_name[person] for person in m.People)
        )

        m.obj = Objective(
            expr=count_if(
                m.Observed[m.person_name[p1], m.person_name[p2], l]
                == m.Expected[p1, p2, l]
                for p1 in m.People
                for p2 in m.People
                for l in m.Languages
            ),
            sense=maximize,
        )

        results = SolverFactory('cp_optimizer').solve(m)

        # we can get one of two perfect matches:
        perfect = 7 * 7 * 4
        self.assertEqual(results.problem.lower_bound, perfect)
        self.assertEqual(results.problem.upper_bound, perfect)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(value(m.obj), perfect)
        self.assertEqual(value(m.person_name['P1']), 0)
        self.assertEqual(value(m.person_name['P2']), 1)
        self.assertEqual(value(m.person_name['P3']), 2)
        self.assertEqual(value(m.person_name['P4']), 3)
        self.assertEqual(value(m.person_name['P5']), 4)
        # We can't distinguish P6 and P7, so they could each have either of
        # names 5 and 6
        self.assertTrue(
            value(m.person_name['P6']) == 5 or value(m.person_name['P6']) == 6
        )
        self.assertTrue(
            value(m.person_name['P7']) == 5 or value(m.person_name['P7']) == 6
        )

        m.person_name['P6'].fix(5)
        m.person_name['P7'].fix(6)

        results = SolverFactory('cp_optimizer').solve(m)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(value(m.obj), perfect)

        m.person_name['P6'].fix(6)
        m.person_name['P7'].fix(5)

        results = SolverFactory('cp_optimizer').solve(m)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(value(m.obj), perfect)

    def test_scheduling_with_sequence_vars(self):
        ct.check_solve_three_step_sequence_model(self, 'cp_optimizer')
