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

class TestGDPoptEnumerate(unittest.TestCase):
    def test_enumerate_gives_correct_answer(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.d = Disjunction(expr=[
            [m.x + m.y >= 5], [m.x - m.y <= 3]
        ])
        m.o = Objective(expr=m.x ** 2)
        results = SolverFactory('gdpopt').solve( m, strategy='enumerate',
                                                 nlp_solver=nlp_solver )
        # correct objective
        self.assertAlmostEqual(value(m.o), 0)
        # correct solution
        self.assertAlmostEqual(value(m.x), 0)
        # y can be anywere between 2 and 3
        self.assertGreaterEqual(value(m.y), 2)
        self.assertLessEqual(value(m.y), 3)
        
        # there were two possibilities to enmerate
        self.assertEqual(results.solver.iterations, 2)
        self.assertIs(results.solver.termination_condition,
                      TerminationCondition.optimal)

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
        # there were two possibilities to enmerate
        self.assertEqual(results.solver.iterations, 2)
        self.assertIs(results.solver.termination_condition,
                      TerminationCondition.optimal)
        
