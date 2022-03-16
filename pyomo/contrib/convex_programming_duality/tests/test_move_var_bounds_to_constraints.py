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
    ConcreteModel, Var, Constraint, NonNegativeReals, NonPositiveReals,
    PositiveIntegers, NegativeIntegers, Block, Binary, NonPositiveReals)
from pyomo.core.base import TransformationFactory
# register the transformation
from pyomo.contrib.convex_programming_duality.\
    move_var_bounds_to_constraints import Move_Variable_Bounds_to_Constraints

class TestMoveVarBoundsToConstraints(unittest.TestCase):
    def test_no_domain_specified(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-3,4))

        TransformationFactory(
            'contrib.move_var_bounds_to_constraints').apply_to(m)

        block = m.component('_pyomo_contrib_var_bounds_constraints')
        self.assertIsInstance(block, Block)
        self.assertEqual(len(block.component_map(Constraint)), 2)
        c1 = block.component('x_lb')
        self.assertIsInstance(c1, Constraint)
        c2 = block.component('x_ub')
        self.assertIsInstance(c2, Constraint)

        self.assertIsNone(m.x.lower)
        self.assertIsNone(m.x.upper)

        self.assertEqual(c1.lower, -3)
        self.assertIsNone(c1.upper)
        self.assertIs(c1.body, m.x)

        self.assertEqual(c2.upper, 4)
        self.assertIsNone(c2.lower)
        self.assertIs(c2.body, m.x)

    def test_lower_bounds_redundant_with_domain_skipped(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 23), domain=NonNegativeReals)
        m.y = Var(bounds=(-40, 3), domain=PositiveIntegers)

        TransformationFactory(
            'contrib.move_var_bounds_to_constraints').apply_to(
                m, bound_constraint_block_name='bounds')
        
        bounds = m.component('bounds')
        self.assertIsInstance(bounds, Block)
        self.assertEqual(len(bounds.component_map(Constraint)), 2)

        # These stay consistent with the domains despite anything, but the point
        # is that we didn't add constraints
        self.assertEqual(m.x.lower, 0)
        self.assertEqual(m.y.lower, 1)
        self.assertIsNone(m.x.upper)
        self.assertIsNone(m.y.upper)

        u1 = bounds.component("x_ub")
        self.assertIsInstance(u1, Constraint)
        self.assertEqual(u1.upper, 23)
        self.assertIsNone(u1.lower)
        self.assertIs(u1.body, m.x)

        u2 = bounds.component("y_ub")
        self.assertIsInstance(u2, Constraint)
        self.assertEqual(u2.upper, 3)
        self.assertIsNone(u2.lower)
        self.assertIs(u2.body, m.y)

    def test_upper_bounds_redundant_with_domain_skipped(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-23, 0), domain=NonPositiveReals)
        m.y = Var(bounds=(-40, 3), domain=Binary)

        TransformationFactory(
            'contrib.move_var_bounds_to_constraints').apply_to(
                m, bound_constraint_block_name='bounds')
        
        bounds = m.component('bounds')
        self.assertIsInstance(bounds, Block)
        m.pprint()
        self.assertEqual(len(bounds.component_map(Constraint)), 1)

        # These stay consistent with the domains despite anything, but the point
        # is that we didn't add constraints
        self.assertEqual(m.x.upper, 0)
        self.assertEqual(m.y.lower, 0)
        self.assertEqual(m.y.upper, 1)
        self.assertIsNone(m.x.lower)

        c = bounds.component('x_lb')
        self.assertIsInstance(c, Constraint)
        self.assertEqual(c.lower, -23)
        self.assertIsNone(c.upper)
        self.assertIs(c.body, m.x)
