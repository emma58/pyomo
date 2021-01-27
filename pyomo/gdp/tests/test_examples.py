#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest

from pyomo.environ import TransformationFactory, SolverFactory, value
import pyomo.opt
from pyomo.gdp.examples.farm_layout.farm_layout import \
    instantiate_model as instantiate_farm_layout

solvers = pyomo.opt.check_available_solvers('bonmin', 'ipopt')

from nose.tools import set_trace

class FarmLayout(unittest.TestCase):
    def check_model_soln(self, m):        
        # check objective value
        self.assertAlmostEqual(value(m.obj), 37.947332, places=4)

        # check solution
        # x
        self.assertAlmostEqual(value(m.x_coord[1]), 0, places=4)
        self.assertAlmostEqual(value(m.x_coord[2]), 0, places=4)

        # y
        self.assertAlmostEqual(value(m.y_coord[1]), 0, places=4)
        self.assertAlmostEqual(value(m.y_coord[2]), 4.2164, places=4)

        # length
        self.assertAlmostEqual(value(m.length[1]), 9.4868, places=4)
        self.assertAlmostEqual(value(m.length[2]), 9.4868, places=4)

        # width
        self.assertAlmostEqual(value(m.width[1]), 4.2164, places=4)
        self.assertAlmostEqual(value(m.width[2]), 5.2705, places=4)

        # total length
        self.assertAlmostEqual(value(m.total_length), 9.4868, places=4)

        # total width
        self.assertAlmostEqual(value(m.total_width), 9.4868, places=4)

        # binaries
        self.assertAlmostEqual(value(m.Above[1,2].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.Below[1,2].indicator_var), 1, places=4)
        self.assertAlmostEqual(value(m.LeftOf[1,2].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.RightOf[1,2].indicator_var), 0, places=4)

    @unittest.skipIf('bonmin' not in solvers, "Bonmin solver not available")
    def test_2_rectangle_bigm(self):
        m = instantiate_farm_layout('farm_layout2.json')

        TransformationFactory('gdp.bigm').apply_to(m)
        results = SolverFactory('bonmin').solve(m)

        self.check_model_soln(m)

    @unittest.skipIf('bonmin' not in solvers, "Bonmin solver not available")
    def test_2_rectangle_cuttingplane(self):
        m = instantiate_farm_layout('farm_layout2.json')

        TransformationFactory('gdp.cuttingplane').apply_to(m)
        results = SolverFactory('bonmin').solve(m)

        self.check_model_soln(m)
        
    @unittest.skipIf('bonmin' not in solvers or 'ipopt' not in solvers, 
                     "Bonmin or Ipopt solver not available")
    def test_2_rectangle_hull(self):
        m = instantiate_farm_layout('farm_layout2.json')

        TransformationFactory('gdp.hull').apply_to(m)
        results = SolverFactory('bonmin').solve(m)

        # This is the same solution, but rotated, so this test has to be his own
        # self.

        # check objective value
        self.assertAlmostEqual(value(m.obj), 37.947332, places=4)

        # check solution
        # x
        self.assertAlmostEqual(value(m.x_coord[1]), 5.2705, places=4)
        self.assertAlmostEqual(value(m.x_coord[2]), 0, places=4)

        # y
        self.assertAlmostEqual(value(m.y_coord[1]), 0, places=4)
        self.assertAlmostEqual(value(m.y_coord[2]), 0, places=4)

        # length
        self.assertAlmostEqual(value(m.length[1]), 4.2164, places=4)
        self.assertAlmostEqual(value(m.length[2]), 5.2705, places=4)

        # width
        self.assertAlmostEqual(value(m.width[1]), 9.4868, places=4)
        self.assertAlmostEqual(value(m.width[2]), 9.4868, places=4)

        # total length
        self.assertAlmostEqual(value(m.total_length), 9.4868, places=4)

        # total width
        self.assertAlmostEqual(value(m.total_width), 9.4868, places=4)

        # binaries
        self.assertAlmostEqual(value(m.Above[1,2].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.Below[1,2].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.LeftOf[1,2].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.RightOf[1,2].indicator_var), 1, places=4)
