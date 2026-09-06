# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
"""Solver-agnostic checks for the models in models.py, parametrized by the
name a CP writer/solver is registered under (e.g. 'cp_optimizer', 'cp_sat').
Each check function solves the model through the named backend and asserts
the same expected results regardless of which backend was used, so a single
check exercises every backend's writer against a shared, once-written model.
"""

from pyomo.contrib.cp.tests import models
from pyomo.environ import SolverFactory, TerminationCondition, value

# For a pure satisfaction (no-objective) model, "solved successfully" isn't
# reported the same way by every backend: CP Optimizer reports `feasible`
# (there being no objective to have proven optimal), while CP-SAT reports
# `optimal` (a feasible solution to a problem with no objective is,
# trivially, optimal). Both mean the same thing here, so checks accept
# either rather than assuming one solver's convention is universal.
_SOLVED = {TerminationCondition.optimal, TerminationCondition.feasible}


def check_solve_mice_and_cookies_model(self, solver_name):
    m = models.mice_and_cookies_model()
    results = SolverFactory(solver_name).solve(m, symbolic_solver_labels=True, tee=True)

    self.assertIn(results.solver.termination_condition, _SOLVED)

    # check solution
    self.assertTrue(value(m.eat_cookie[0].is_present))
    self.assertTrue(value(m.eat_cookie[1].is_present))
    # That means there were crumbs:
    self.assertEqual(value(m.num_crumbs), 5)
    # So there was sweeping:
    self.assertTrue(value(m.sweep_crumbs.is_present))

    # start with the first cookie:
    self.assertEqual(value(m.eat_cookie[0].start_time), 0)
    self.assertEqual(value(m.eat_cookie[0].end_time), 8)
    self.assertEqual(value(m.eat_cookie[0].length), 8)
    # Proceed to second cookie:
    self.assertEqual(value(m.eat_cookie[1].start_time), 8)
    self.assertEqual(value(m.eat_cookie[1].end_time), 16)
    self.assertEqual(value(m.eat_cookie[1].length), 8)
    # Sweep
    self.assertEqual(value(m.sweep_crumbs.start_time), 16)
    self.assertEqual(value(m.sweep_crumbs.end_time), 17)
    self.assertEqual(value(m.sweep_crumbs.length), 1)
    # End with read story, as it keeps exactly one mouse occupied
    # indefinitely (in this particular retelling)
    self.assertEqual(value(m.read_story.start_time), 17)

    # Since doing the dishes actually *bores* a mouse, we leave the dishes
    # in the sink
    self.assertFalse(value(m.do_dishes.is_present))

    self.assertEqual(results.problem.number_of_objectives, 0)

    return results


def check_solve_three_step_sequence_model(self, solver_name):
    m = models.three_step_sequence_model()

    results = SolverFactory(solver_name).solve(m)
    self.assertIn(results.solver.termination_condition, _SOLVED)
    self.assertEqual(value(m.i[1].start_time), 0)
    self.assertEqual(value(m.i[2].start_time), 2)
    self.assertEqual(value(m.i[3].start_time), 6)

    return results
