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

from pyomo.environ import (TransformationFactory, SolverFactory, value,
                           TerminationCondition)
import pyomo.opt
from pyomo.gdp.examples.farm_layout.farm_layout import \
    instantiate_model as instantiate_farm_layout
from pyomo.gdp.examples.safety_layout.safety_layout import \
    instantiate_model as instantiate_safety_layout

from os.path import abspath, dirname, normpath, join
currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir,'..','examples'))

solvers = pyomo.opt.check_available_solvers('bonmin', 'ipopt', 'baron',
                                            'gurobi')

from nose.tools import set_trace

class FarmLayout(unittest.TestCase):
    # ESJ: This problem has some symmetry: You can rotate your farm in one of
    # four ways. To make these tests less fragile, we solve the MINLP and check
    # the objective value. Then we fix the binaries to a solution we know, solve
    # the NLP and check the continuous variables.
    def check_2_rectangle_model_soln(self, m):        
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

    @unittest.skipIf('bonmin' not in solvers or 'ipopt' not in solvers, 
                     "Bonmin or Ipopt solver not available")
    def test_2_rectangle_bigm(self):
        m = instantiate_farm_layout(join(exdir, 'farm_layout',
                                         'farm_layout2.json'))

        TransformationFactory('gdp.bigm').apply_to(m)
        results = SolverFactory('bonmin').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.obj), 37.947332, places=4)

        # fix the binaries to a known solution
        m.Above[1,2].indicator_var.fix(0)
        m.Below[1,2].indicator_var.fix(1)
        m.LeftOf[1,2].indicator_var.fix(0)
        m.RightOf[1,2].indicator_var.fix(0)

        results = SolverFactory('ipopt').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)

        self.check_2_rectangle_model_soln(m)

    @unittest.skipIf('bonmin' not in solvers or 'ipopt' not in solvers, 
                     "Bonmin or Ipopt solver not available")
    def test_2_rectangle_cuttingplane(self):
        m = instantiate_farm_layout(join(exdir,'farm_layout',
                                         'farm_layout2.json'))

        TransformationFactory('gdp.cuttingplane').apply_to(m)
        results = SolverFactory('bonmin').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.obj), 37.947332, places=4)

        # fix the binaries to a known solution
        m.Above[1,2].indicator_var.fix(0)
        m.Below[1,2].indicator_var.fix(1)
        m.LeftOf[1,2].indicator_var.fix(0)
        m.RightOf[1,2].indicator_var.fix(0)

        results = SolverFactory('ipopt').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)

        self.check_2_rectangle_model_soln(m)

    @unittest.skipIf('bonmin' not in solvers or 'ipopt' not in solvers, 
                     "Bonmin or Ipopt solver not available")
    def test_2_rectangle_hull(self):
        m = instantiate_farm_layout(join(exdir,'farm_layout',
                                         'farm_layout2.json'))

        TransformationFactory('gdp.hull').apply_to(m)
        results = SolverFactory('bonmin').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.obj), 37.947332, places=4)

        # fix the binaries to a known solution
        m.Above[1,2].indicator_var.fix(0)
        m.Below[1,2].indicator_var.fix(1)
        m.LeftOf[1,2].indicator_var.fix(0)
        m.RightOf[1,2].indicator_var.fix(0)

        results = SolverFactory('ipopt').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)

        self.check_2_rectangle_model_soln(m)

    def check_3_rectangle_model_soln(self, m):
        # check objective value
        self.assertAlmostEqual(value(m.obj), 48.9898, places=4)

        # check solution
        # x
        self.assertAlmostEqual(value(m.x_coord[1]), 0, places=4)
        self.assertAlmostEqual(value(m.x_coord[2]), 0, places=4)
        self.assertAlmostEqual(value(m.x_coord[3]), 4.8990, places=4)

        # y
        self.assertAlmostEqual(value(m.y_coord[1]), 0, places=4)
        self.assertAlmostEqual(value(m.y_coord[2]), 8.1650, places=4)
        self.assertAlmostEqual(value(m.y_coord[3]), 0, places=4)

        # length
        self.assertAlmostEqual(value(m.length[1]), 4.8990, places=4)
        self.assertAlmostEqual(value(m.length[2]), 12.2474, places=4)
        self.assertAlmostEqual(value(m.length[3]), 7.3485, places=4)

        # width
        self.assertAlmostEqual(value(m.width[1]), 8.1650, places=4)
        self.assertAlmostEqual(value(m.width[2]), 4.0825, places=4)
        self.assertAlmostEqual(value(m.width[3]), 8.1650, places=4)

        # total length
        self.assertAlmostEqual(value(m.total_length), 12.2474, places=4)

        # total width
        self.assertAlmostEqual(value(m.total_width), 12.2474, places=4)

    @unittest.category('expensive')
    @unittest.skipIf('bonmin' not in solvers or 'ipopt' not in solvers,
                     "Bonmin or Ipopt solver not available")
    def test_3_rectangle_cuttingplane(self):
        m = instantiate_farm_layout(join(exdir, 'farm_layout',
                                         'farm_layout3.json'))

        TransformationFactory('gdp.cuttingplane').apply_to(m)
        results = SolverFactory('bonmin').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.obj), 48.9898, places=4)

        # fix the binaries to a known solution
        m.Below[1,2].indicator_var.fix(1)
        m.Above[1,2].indicator_var.fix(0)
        m.LeftOf[1,2].indicator_var.fix(0)
        m.RightOf[1,2].indicator_var.fix(0)

        m.LeftOf[1,3].indicator_var.fix(1)
        m.RightOf[1,3].indicator_var.fix(0)
        m.Above[1,3].indicator_var.fix(0)
        m.Below[1,3].indicator_var.fix(0)

        m.Above[2,3].indicator_var.fix(1)
        m.Below[2,3].indicator_var.fix(0)
        m.RightOf[2,3].indicator_var.fix(0)
        m.LeftOf[2,3].indicator_var.fix(0)

        results = SolverFactory('ipopt').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)

        self.check_3_rectangle_model_soln(m)

    @unittest.category('expensive')
    @unittest.skipIf('bonmin' not in solvers or 'ipopt' not in solvers, 
                     "Bonmin or Ipopt solver not available")
    def test_3_rectangle_bigm(self):
        m = instantiate_farm_layout(join(exdir, 'farm_layout',
                                         'farm_layout3.json'))

        TransformationFactory('gdp.bigm').apply_to(m)
        results = SolverFactory('bonmin').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.obj), 48.9898, places=4)

        # fix the binaries to a known solution
        m.Below[1,2].indicator_var.fix(1)
        m.Above[1,2].indicator_var.fix(0)
        m.LeftOf[1,2].indicator_var.fix(0)
        m.RightOf[1,2].indicator_var.fix(0)

        m.LeftOf[1,3].indicator_var.fix(1)
        m.RightOf[1,3].indicator_var.fix(0)
        m.Above[1,3].indicator_var.fix(0)
        m.Below[1,3].indicator_var.fix(0)

        m.Above[2,3].indicator_var.fix(1)
        m.Below[2,3].indicator_var.fix(0)
        m.RightOf[2,3].indicator_var.fix(0)
        m.LeftOf[2,3].indicator_var.fix(0)

        results = SolverFactory('ipopt').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        
        self.check_3_rectangle_model_soln(m)

    @unittest.category('expensive')
    @unittest.skipIf('bonmin' not in solvers or 'ipopt' not in solvers, 
                     "Bonmin or Ipopt solver not available")
    def test_3_rectangle_hull(self):
        m = instantiate_farm_layout(join(exdir, 'farm_layout',
                                         'farm_layout3.json'))

        TransformationFactory('gdp.hull').apply_to(m)
        results = SolverFactory('bonmin').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.obj), 48.9898, places=4)

        # fix the binaries to a known solution
        m.Below[1,2].indicator_var.fix(1)
        m.Above[1,2].indicator_var.fix(0)
        m.LeftOf[1,2].indicator_var.fix(0)
        m.RightOf[1,2].indicator_var.fix(0)

        m.LeftOf[1,3].indicator_var.fix(1)
        m.RightOf[1,3].indicator_var.fix(0)
        m.Above[1,3].indicator_var.fix(0)
        m.Below[1,3].indicator_var.fix(0)

        m.Above[2,3].indicator_var.fix(1)
        m.Below[2,3].indicator_var.fix(0)
        m.RightOf[2,3].indicator_var.fix(0)
        m.LeftOf[2,3].indicator_var.fix(0)

        results = SolverFactory('ipopt').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)

        self.check_3_rectangle_model_soln(m)

    def check_4_rectangle_model_soln(self, m):
        # check objective value
        self.assertAlmostEqual(value(m.obj), 54.405882, places=4)

        # check solution
        # x
        self.assertAlmostEqual(value(m.x_coord[1]), 0, places=4)
        self.assertAlmostEqual(value(m.x_coord[2]), 9.9254, places=4)
        self.assertAlmostEqual(value(m.x_coord[3]), 3.6567, places=4)
        self.assertAlmostEqual(value(m.x_coord[4]), 0, places=4)

        # y
        self.assertAlmostEqual(value(m.y_coord[1]), 9.5714, places=4)
        self.assertAlmostEqual(value(m.y_coord[2]), 0, places=4)
        self.assertAlmostEqual(value(m.y_coord[3]), 0, places=4)
        self.assertAlmostEqual(value(m.y_coord[4]), 0, places=4)

        # length
        self.assertAlmostEqual(value(m.length[1]), 9.9254, places=4)
        self.assertAlmostEqual(value(m.length[2]), 3.6761, places=4)
        self.assertAlmostEqual(value(m.length[3]), 6.2687, places=4)
        self.assertAlmostEqual(value(m.length[4]), 3.6567, places=4)

        # width
        self.assertAlmostEqual(value(m.width[1]), 4.0301, places=4)
        self.assertAlmostEqual(value(m.width[2]), 13.6015, places=4)
        self.assertAlmostEqual(value(m.width[3]), 9.5714, places=4)
        self.assertAlmostEqual(value(m.width[4]), 9.5714, places=4)

        # total length
        self.assertAlmostEqual(value(m.total_length), 13.6015, places=4)

        # total width
        self.assertAlmostEqual(value(m.total_width), 13.6015, places=4)

    @unittest.category('expensive')
    @unittest.skipIf('baron' not in solvers or 'ipopt' not in solvers,
                     "Baron or Ipopt solver not available")
    def test_4_rectangle_cuttingplane(self):
        m = instantiate_farm_layout(join(exdir, 'farm_layout',
                                         'farm_layout4.json'))

        TransformationFactory('gdp.cuttingplane').apply_to(m)
        results = SolverFactory('baron').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.obj), 54.4059, places=4)

        # fix the binaries to a known solution
        m.LeftOf[1,2].indicator_var.fix(1)
        m.Above[1,2].indicator_var.fix(0)
        m.Below[1,2].indicator_var.fix(0)
        m.RightOf[1,2].indicator_var.fix(0)

        m.Above[1,3].indicator_var.fix(1)
        m.RightOf[1,3].indicator_var.fix(0)
        m.LeftOf[1,3].indicator_var.fix(0)
        m.Below[1,3].indicator_var.fix(0)

        m.RightOf[2,3].indicator_var.fix(1)
        m.Below[2,3].indicator_var.fix(0)
        m.Above[2,3].indicator_var.fix(0)
        m.LeftOf[2,3].indicator_var.fix(0)

        m.Above[1,4].indicator_var.fix(1)
        m.Below[1,4].indicator_var.fix(0)
        m.RightOf[1,4].indicator_var.fix(0)
        m.LeftOf[1,4].indicator_var.fix(0)

        m.RightOf[2,4].indicator_var.fix(1)
        m.Below[2,4].indicator_var.fix(0)
        m.Above[2,4].indicator_var.fix(0)
        m.LeftOf[2,4].indicator_var.fix(0)

        m.RightOf[3,4].indicator_var.fix(1)
        m.Below[3,4].indicator_var.fix(0)
        m.Above[3,4].indicator_var.fix(0)
        m.LeftOf[3,4].indicator_var.fix(0)

        results = SolverFactory('ipopt').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)

        self.check_4_rectangle_model_soln(m)

    @unittest.category('expensive')
    @unittest.skipIf('baron' not in solvers or 'ipopt' not in solvers, 
                     "Baron or Ipopt solver not available")
    def test_4_rectangle_bigm(self):
        m = instantiate_farm_layout(join(exdir, 'farm_layout',
                                         'farm_layout4.json'))

        TransformationFactory('gdp.bigm').apply_to(m)
        results = SolverFactory('baron').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.obj), 54.4058, places=4)

        # fix the binaries to a known solution
        m.LeftOf[1,2].indicator_var.fix(1)
        m.Above[1,2].indicator_var.fix(0)
        m.Below[1,2].indicator_var.fix(0)
        m.RightOf[1,2].indicator_var.fix(0)

        m.Above[1,3].indicator_var.fix(1)
        m.RightOf[1,3].indicator_var.fix(0)
        m.LeftOf[1,3].indicator_var.fix(0)
        m.Below[1,3].indicator_var.fix(0)

        m.RightOf[2,3].indicator_var.fix(1)
        m.Below[2,3].indicator_var.fix(0)
        m.Above[2,3].indicator_var.fix(0)
        m.LeftOf[2,3].indicator_var.fix(0)

        m.Above[1,4].indicator_var.fix(1)
        m.Below[1,4].indicator_var.fix(0)
        m.RightOf[1,4].indicator_var.fix(0)
        m.LeftOf[1,4].indicator_var.fix(0)

        m.RightOf[2,4].indicator_var.fix(1)
        m.Below[2,4].indicator_var.fix(0)
        m.Above[2,4].indicator_var.fix(0)
        m.LeftOf[2,4].indicator_var.fix(0)

        m.RightOf[3,4].indicator_var.fix(1)
        m.Below[3,4].indicator_var.fix(0)
        m.Above[3,4].indicator_var.fix(0)
        m.LeftOf[3,4].indicator_var.fix(0)

        results = SolverFactory('ipopt').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        
        self.check_4_rectangle_model_soln(m)

    @unittest.category('expensive')
    @unittest.skipIf('baron' not in solvers or 'ipopt' not in solvers, 
                     "Baron or Ipopt solver not available")
    def test_4_rectangle_hull(self):
        m = instantiate_farm_layout(join(exdir, 'farm_layout',
                                         'farm_layout4.json'))

        TransformationFactory('gdp.hull').apply_to(m)
        results = SolverFactory('baron').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.obj), 54.4058, places=4)

        # fix the binaries to a known solution
        m.LeftOf[1,2].indicator_var.fix(1)
        m.Above[1,2].indicator_var.fix(0)
        m.Below[1,2].indicator_var.fix(0)
        m.RightOf[1,2].indicator_var.fix(0)

        m.Above[1,3].indicator_var.fix(1)
        m.RightOf[1,3].indicator_var.fix(0)
        m.LeftOf[1,3].indicator_var.fix(0)
        m.Below[1,3].indicator_var.fix(0)

        m.RightOf[2,3].indicator_var.fix(1)
        m.Below[2,3].indicator_var.fix(0)
        m.Above[2,3].indicator_var.fix(0)
        m.LeftOf[2,3].indicator_var.fix(0)

        m.Above[1,4].indicator_var.fix(1)
        m.Below[1,4].indicator_var.fix(0)
        m.RightOf[1,4].indicator_var.fix(0)
        m.LeftOf[1,4].indicator_var.fix(0)

        m.RightOf[2,4].indicator_var.fix(1)
        m.Below[2,4].indicator_var.fix(0)
        m.Above[2,4].indicator_var.fix(0)
        m.LeftOf[2,4].indicator_var.fix(0)

        m.RightOf[3,4].indicator_var.fix(1)
        m.Below[3,4].indicator_var.fix(0)
        m.Above[3,4].indicator_var.fix(0)
        m.LeftOf[3,4].indicator_var.fix(0)

        results = SolverFactory('ipopt').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)

        self.check_4_rectangle_model_soln(m)

    def check_5_rectangle_model_soln(self, m):
        # check objective value
        self.assertAlmostEqual(value(m.obj), TODO, places=4)

        # check solution
        # x
        self.assertAlmostEqual(value(m.x_coord[1]), 0, places=4)
        self.assertAlmostEqual(value(m.x_coord[2]), 9.9254, places=4)
        self.assertAlmostEqual(value(m.x_coord[3]), 3.6567, places=4)
        self.assertAlmostEqual(value(m.x_coord[4]), 0, places=4)

        # y
        self.assertAlmostEqual(value(m.y_coord[1]), 9.5714, places=4)
        self.assertAlmostEqual(value(m.y_coord[2]), 0, places=4)
        self.assertAlmostEqual(value(m.y_coord[3]), 0, places=4)
        self.assertAlmostEqual(value(m.y_coord[4]), 0, places=4)

        # length
        self.assertAlmostEqual(value(m.length[1]), 9.9254, places=4)
        self.assertAlmostEqual(value(m.length[2]), 3.6761, places=4)
        self.assertAlmostEqual(value(m.length[3]), 6.2687, places=4)
        self.assertAlmostEqual(value(m.length[4]), 3.6567, places=4)

        # width
        self.assertAlmostEqual(value(m.width[1]), 4.0301, places=4)
        self.assertAlmostEqual(value(m.width[2]), 13.6015, places=4)
        self.assertAlmostEqual(value(m.width[3]), 9.5714, places=4)
        self.assertAlmostEqual(value(m.width[4]), 9.5714, places=4)

        # total length
        self.assertAlmostEqual(value(m.total_length), 13.6015, places=4)

        # total width
        self.assertAlmostEqual(value(m.total_width), 13.6015, places=4)

    @unittest.skipIf('ipopt' not in solvers, "Ipopt solver not available")
    def test_5_rectangle_bigm(self):
        m = instantiate_farm_layout(join(exdir, 'farm_layout',
                                         'farm_layout5.json'))

        TransformationFactory('gdp.bigm').apply_to(m)
        # TODO: This takes baron 800 seconds... Should we just fix the binaries
        # and solve to test this and the 6-rectangle problem??
        results = SolverFactory('baron').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        set_trace()
        self.assertAlmostEqual(value(m.obj), 64.4976, places=4)

        # fix the binaries to a known solution

class SafetyLayout(unittest.TestCase):
    def check_4_rectangle_soln(self, m):
        self.assertAlmostEqual(value(m.x[1]), 3.6353, places=4)
        self.assertAlmostEqual(value(m.x[2]), 9.6353, places=4)
        self.assertAlmostEqual(value(m.x[3]), 7.6353, places=4)
        self.assertAlmostEqual(value(m.x[4]), 3.6353, places=4)

        self.assertAlmostEqual(value(m.y[1]), 9.9545, places=4)
        self.assertAlmostEqual(value(m.y[2]), 14.2949, places=4)
        self.assertAlmostEqual(value(m.y[3]), 9.4583, places=4)
        self.assertAlmostEqual(value(m.y[4]), 5.4545, places=4)

        self.assertAlmostEqual(value(m.Above[1,2].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.Below[1,2].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.LeftOf[1,2].indicator_var), 1, places=4)
        self.assertAlmostEqual(value(m.RightOf[1,2].indicator_var), 0, places=4)

        self.assertAlmostEqual(value(m.Above[1,3].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.Below[1,3].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.LeftOf[1,3].indicator_var), 1, places=4)
        self.assertAlmostEqual(value(m.RightOf[1,3].indicator_var), 0, places=4)

        self.assertAlmostEqual(value(m.Above[1,4].indicator_var), 1, places=4)
        self.assertAlmostEqual(value(m.Below[1,4].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.LeftOf[1,4].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.RightOf[1,4].indicator_var), 0, places=4)

        self.assertAlmostEqual(value(m.Above[2,3].indicator_var), 1, places=4)
        self.assertAlmostEqual(value(m.Below[2,3].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.LeftOf[2,3].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.RightOf[2,3].indicator_var), 0, places=4)

        self.assertAlmostEqual(value(m.Above[2,4].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.Below[2,4].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.LeftOf[2,4].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.RightOf[2,4].indicator_var), 1, places=4)

        self.assertAlmostEqual(value(m.Above[3,4].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.Below[3,4].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.LeftOf[3,4].indicator_var), 0, places=4)
        self.assertAlmostEqual(value(m.RightOf[3,4].indicator_var), 1, places=4)

        self.assertAlmostEqual(value(m.x_dist[1,2]), 6, places=4)
        self.assertAlmostEqual(value(m.y_dist[1,2]), 4.3403, places=4)
        self.assertAlmostEqual(value(m.x_dist[1,3]), 4, places=4)
        self.assertAlmostEqual(value(m.y_dist[1,3]), 0.4962, places=4)
        self.assertAlmostEqual(value(m.x_dist[1,4]), 0, places=4)
        self.assertAlmostEqual(value(m.y_dist[1,4]), 4.5, places=4)
        self.assertAlmostEqual(value(m.x_dist[2,3]), 2, places=4)
        self.assertAlmostEqual(value(m.y_dist[2,3]), 4.8365, places=4)
        self.assertAlmostEqual(value(m.x_dist[2,4]), 6, places=4)
        self.assertAlmostEqual(value(m.y_dist[2,4]), 8.8403, places=4)
        self.assertAlmostEqual(value(m.x_dist[3,4]), 4, places=4)
        self.assertAlmostEqual(value(m.y_dist[3,4]), 4.0038, places=4)

    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
    def test_4_rectangle_bigm(self):
        m = instantiate_safety_layout(join(exdir, 'safety_layout', 
                                           'safety_layout4.json'))
        TransformationFactory('gdp.bigm').apply_to(m)
        results = SolverFactory('gurobi').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.obj), 9859.659708, places=4)

        self.check_4_rectangle_soln(m)
        
    @unittest.skipIf('gurobi' not in solvers, "Gurobi solver not available")
    def test_4_rectangle_hull(self):
        m = instantiate_safety_layout(join(exdir, 'safety_layout', 
                                           'safety_layout4.json'))
        TransformationFactory('gdp.hull').apply_to(m)
        results = SolverFactory('gurobi').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.obj), 9859.659708, places=4)

        self.check_4_rectangle_soln(m)

    @unittest.skipIf('gurobi' not in solvers or 'ipopt' not in solvers, 
                     "Gurobi or Ipopt solver not available")
    def test_4_rectangle_cuttingplane(self):
        m = instantiate_safety_layout(join(exdir, 'safety_layout', 
                                           'safety_layout4.json'))
        TransformationFactory('gdp.cuttingplane').apply_to(m)
        results = SolverFactory('gurobi').solve(m)
        self.assertTrue(results.solver.termination_condition == \
                        TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.obj), 9859.659708, places=4)

        self.check_4_rectangle_soln(m)
