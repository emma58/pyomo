#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.gdp.tests import models
from pyomo.contrib.gdpopt.GDPopt import GDPoptSolver
from pyomo.environ import (ConcreteModel, Var, Objective, value, SolverFactory,
                           maximize)
from pyomo.opt import TerminationCondition

from pyomo.gdp import (Disjunction)

nlp_solver = 'ipopt'

@unittest.skipUnless(SolverFactory(nlp_solver).available(),
                     "NLP solver not available")
class TestGDPoptEnumerate(unittest.TestCase):
    def test_linear_model_quadratic_objective(self):
        m = models.makeTwoTermLinearWithQuadraticObj()
        results = SolverFactory('gdpopt').solve( m, strategy='enumerate',
                                                 nlp_solver=nlp_solver )
        # correct objective
        self.assertAlmostEqual(value(m.o), 0)
        # correct solution
        self.assertAlmostEqual(value(m.x), 0)
        # y can be anywere between 2 and 3
        self.assertGreaterEqual(value(m.y), 2)
        self.assertLessEqual(value(m.y), 3)

        # there were two possibilities to enumerate
        self.assertEqual(results.solver.iterations, 2)
        self.assertIs(results.solver.termination_condition,
                      TerminationCondition.optimal)
        self.assertAlmostEqual(results.problem.lower_bound, 0)
        self.assertAlmostEqual(results.problem.upper_bound, 0)

        m.o.sense = maximize
        results = SolverFactory('gdpopt').solve( m, strategy='enumerate',
                                                 nlp_solver=nlp_solver )
        # correct objective
        self.assertAlmostEqual(value(m.o), 100)
        # correct solution
        self.assertAlmostEqual(value(m.x), 10)
        # y can be anywere between 2 and 3
        self.assertGreaterEqual(value(m.y), 2)
        self.assertLessEqual(value(m.y), 3)

        print(results)
        # there were two possibilities to enumerate
        self.assertEqual(results.solver.iterations, 2)
        self.assertIs(results.solver.termination_condition,
                      TerminationCondition.optimal)
        self.assertAlmostEqual(results.problem.lower_bound, 100)
        self.assertAlmostEqual(results.problem.upper_bound, 100)

    def test_unbounded_model(self):
        pass

    def test_infeasible_model(self):
        # This model is infeasible because both Disjuncts violate the bounds of
        # x. So we'll find out in the subproblems.
        m = models.makeInfeasibleModel()
        results = SolverFactory('gdpopt').solve( m, strategy='enumerate',
                                                 nlp_solver=nlp_solver )
        self.assertIs(results.solver.termination_condition,
                      TerminationCondition.infeasible)
        # should have tried both subproblems because the discrete solutions are
        # both feasible.
        self.assertEqual(results.solver.iterations, 2)
        self.assertEqual(results.problem.lower_bound, float('inf'))
        self.assertEqual(results.problem.upper_bound, float('inf'))

        self.assertIsNone(m.x.value)
        self.assertIsNone(m.d.disjuncts[0].indicator_var.value)
        self.assertIsNone(m.d.disjuncts[1].indicator_var.value)

    def test_integer_infeasible_model(self):
        # This model is infeasible because there aren't actually any feasible
        # discrete solutions.
        pass

    def test_nested_disjunctive_model(self):
        pass

    def test_discrete_vars_on_disjuncts(self):
        pass

    def test_non_indicator_boolean_vars(self):
        pass
