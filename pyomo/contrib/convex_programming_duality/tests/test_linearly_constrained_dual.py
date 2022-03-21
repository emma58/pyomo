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
from pyomo.core import (
    ConcreteModel, Var, Constraint, NonNegativeReals, NonPositiveReals, Block,
    Reals, NonNegativeReals, Objective, value, maximize, Integers)
from pyomo.core.base import TransformationFactory
# register the transformation
from pyomo.contrib.convex_programming_duality.linearly_constrained_dual import \
    Linearly_Constrained_Dual
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import check_linear_coef
from pytest import set_trace

class TestLinearlyConstrainedDual(unittest.TestCase):
    def check_transformation_block(self, blk):
        self.assertIsInstance(blk, Block)

        self.assertEqual(len(blk.component_map(Var)), 3)
        d1 = blk.component("equality_dual")
        self.assertIsInstance(d1, Var)
        self.assertEqual(d1.domain, Reals)
        d2 = blk.component("geq_dual")
        self.assertIsInstance(d2, Var)
        self.assertEqual(d2.domain, NonNegativeReals)
        d3 = blk.component("leq_dual")
        self.assertIsInstance(d3, Var)
        self.assertEqual(d3.domain, NonPositiveReals)

        self.assertEqual(len(blk.component_map(Constraint)), 2)
        c1 = blk.component("x")
        self.assertIsInstance(c1, Constraint)
        self.assertEqual(value(c1.lower), 2)
        self.assertEqual(value(c1.upper), 2)
        repn = generate_standard_repn(c1.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 3)
        check_linear_coef(self, repn, d1, 4)
        check_linear_coef(self, repn, d2, 1)
        check_linear_coef(self, repn, d3, 1)
        self.assertEqual(repn.constant, 0)

        c2 = blk.component("y")
        self.assertIsInstance(c2, Constraint)
        self.assertEqual(c2.upper, 3)
        self.assertIsNone(c2.lower)
        repn = generate_standard_repn(c2.body)
        self.assertTrue(repn.is_linear())
        self.assertEqual(repn.constant, 0)
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, d1, 1)
        check_linear_coef(self, repn, d3, -1)
        
        self.assertEqual(len(blk.component_map(Objective)), 1)
        obj = blk.component("dual_obj")
        self.assertIsInstance(obj, Objective)
        self.assertEqual(obj.sense, maximize)
        repn = generate_standard_repn(obj.expr)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        check_linear_coef(self, repn, d1, 6)
        check_linear_coef(self, repn, d2, 0.2)
        self.assertEqual(repn.constant, 0)

    def check_original_components_deactivated(self, m):
        self.assertFalse(m.obj.active)
        self.assertFalse(m.equality.active)
        self.assertFalse(m.geq.active)
        self.assertFalse(m.leq.active)

    def make_minimization_model(self):
        m = ConcreteModel()
        m.x = Var(domain=Reals)
        m.y = Var(domain=NonNegativeReals)
        m.obj = Objective(expr=2*m.x + 3*m.y)
        m.equality = Constraint(expr=4*m.x + m.y == 6)
        m.geq = Constraint(expr=m.x >= 0.2)
        m.leq = Constraint(expr=m.x - m.y <= 0)

        return m

    def test_minimization_dual(self):
        m = self.make_minimization_model()

        TransformationFactory(
            'contrib.convex_linearly_constrained_dual').apply_to(m)

        blk = m.component('_pyomo_contrib_linearly_constrained_dual')
        self.check_transformation_block(blk)
        self.check_original_components_deactivated(m)
        
    def test_maximization_dual(self):
        pass

    def test_var_bounds_that_are_constraints_get_dual_vars(self):
        pass

    def test_fixed_vars_fixed_forever(self):
        m = ConcreteModel()
        m.x = Var(domain=Reals)
        m.y = Var(domain=NonNegativeReals)
        m.jk = Var(domain=Integers)
        m.jk.fix(6)
        m.obj = Objective(expr=2*m.x + (m.jk/2)*m.y)
        m.equality = Constraint(expr=4*m.x + m.y == m.jk)
        m.geq = Constraint(expr=m.x >= 0.2)
        m.leq = Constraint(expr=m.x - m.y  + m.jk <= 6)

        TransformationFactory(
            'contrib.convex_linearly_constrained_dual').apply_to(
                m, assume_fixed_vars_permanent=True)

        blk = m.component('_pyomo_contrib_linearly_constrained_dual')
        self.check_transformation_block(blk)
        self.check_original_components_deactivated(m)

    def test_fixed_vars_not_fixed_forever(self):
        m = self.make_minimization_model()
        m.x.fix(13)

        TransformationFactory(
            'contrib.convex_linearly_constrained_dual').apply_to(
                m, assume_fixed_vars_permanent=False)

        blk = m.component('_pyomo_contrib_linearly_constrained_dual')
        self.check_transformation_block(blk)
        self.check_original_components_deactivated(m)

    def test_vars_treated_as_rhs(self):
        pass

    def test_indexed_constraints_make_indexed_dual_vars(self):
        pass

    def test_indexed_vars_make_indexed_dual_constraints(self):
        m = ConcreteModel()
        m.x = Var([1, 2, 3], domain=NonNegativeReals)
        m.obj = Objective(expr=m.x[1] + m.x[2])
        m.cons = Constraint(expr=m.x[1] + 2*m.x[2] + 3*m.x[3] >= 7)

        TransformationFactory(
            'contrib.convex_linearly_constrained_dual').apply_to(m)

        blk = m.component('_pyomo_contrib_linearly_constrained_dual')
        self.assertEqual(len(blk.component_map(Var)), 1)
        self.assertEqual(len(blk.component_map(Constraint)), 1)
        self.assertEqual(len(blk.component_map(Objective)), 1)

        d = blk.component("cons_dual")
        self.assertIsInstance(d, Var)
        self.assertEqual(d.domain, NonNegativeReals)
        dual_constraints = blk.component("x")
        self.assertIsInstance(dual_constraints, Constraint)
        self.assertEqual(len(dual_constraints), 3)

        c1 = dual_constraints[1]
        self.assertEqual(c1.upper, 1)
        self.assertIsNone(c1.lower)
        repn = generate_standard_repn(c1.body)
        self.assertEqual(repn.constant, 0)
        check_linear_coef(self, repn, d, 1)
        self.assertEqual(len(repn.linear_vars), 1)

        c2 = dual_constraints[2]
        self.assertEqual(c2.upper, 1)
        self.assertIsNone(c2.lower)
        repn = generate_standard_repn(c2.body)
        self.assertEqual(repn.constant, 0)
        check_linear_coef(self, repn, d, 2)
        self.assertEqual(len(repn.linear_vars), 1)

        c3 = dual_constraints[3]
        self.assertEqual(c3.upper, 0)
        self.assertIsNone(c2.lower)
        repn = generate_standard_repn(c3.body)
        self.assertEqual(repn.constant, 0)
        check_linear_coef(self, repn, d, 3)
        self.assertEqual(len(repn.linear_vars), 1)

        obj = blk.component("dual_obj")
        self.assertIsInstance(obj, Objective)
        repn = generate_standard_repn(obj.expr)
        self.assertEqual(repn.constant, 0)
        check_linear_coef(self, repn, d, 7)
        self.assertEqual(len(repn.linear_vars), 1)

    def test_deactivated_primal_constraints_ignored(self):
        m = self.make_minimization_model()
        m.silly_cons = Constraint(expr=m.x + 45*m.y == 9)
        m.silly_cons.deactivate()

        TransformationFactory(
            'contrib.convex_linearly_constrained_dual').apply_to(m)

        blk = m.component('_pyomo_contrib_linearly_constrained_dual')
        self.check_transformation_block(blk)
        self.check_original_components_deactivated(m)

        self.assertFalse(m.silly_cons.active)
