# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

#
# Unit Tests for Integral Objects
#

import os
from os.path import abspath, dirname

import pyomo.common.unittest as unittest

from pyomo.environ import ConcreteModel, Var, Set, TransformationFactory, Expression
from pyomo.dae import ContinuousSet, Integral
from pyomo.dae.diffvar import DAE_Error

from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import OrderedVarRecorder

currdir = dirname(abspath(__file__)) + os.sep


class TestIntegral(unittest.TestCase):
    # test valid declarations
    def test_valid(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 1))
        m.x = ContinuousSet(bounds=(5, 10))
        m.s = Set(initialize=[1, 2, 3])
        m.v = Var(m.t)
        m.v2 = Var(m.s, m.t)
        m.v3 = Var(m.t, m.x)

        def _int1(m, t):
            return m.v[t]

        m.int1 = Integral(m.t, rule=_int1)

        def _int2(m, s, t):
            return m.v2[s, t]

        m.int2 = Integral(m.s, m.t, wrt=m.t, rule=_int2)

        def _int3(m, t, x):
            return m.v3[t, x]

        m.int3 = Integral(m.t, m.x, wrt=m.t, rule=_int3)

        def _int4(m, x):
            return m.int3[x]

        m.int4 = Integral(m.x, wrt=m.x, rule=_int4)

        self.assertTrue(isinstance(m.int1, Expression))
        self.assertTrue(isinstance(m.int2, Expression))
        self.assertTrue(isinstance(m.int3, Expression))
        self.assertTrue(isinstance(m.int4, Expression))
        self.assertTrue(m.int1.get_continuousset() is m.t)
        self.assertTrue(m.int2.get_continuousset() is m.t)
        self.assertTrue(m.int3.get_continuousset() is m.t)
        self.assertTrue(m.int4.get_continuousset() is m.x)
        self.assertEqual(len(m.int1), 1)
        self.assertEqual(len(m.int2), 3)
        self.assertEqual(len(m.int3), 2)
        self.assertEqual(len(m.int4), 1)
        self.assertTrue(m.int1.ctype is Integral)
        self.assertTrue(m.int2.ctype is Integral)
        self.assertTrue(m.int3.ctype is Integral)
        self.assertTrue(m.int4.ctype is Integral)

        visitor = LinearRepnVisitor({}, var_recorder=OrderedVarRecorder({}, {}, None))

        repn = visitor.walk_expression(m.int1.expr)
        self.assertEqual(len(repn.linear), 2)
        self.assertIn(id(m.v[1]), repn.linear)
        self.assertEqual(repn.linear[id(m.v[1])], 0.5)
        self.assertIn(id(m.v[0]), repn.linear)
        self.assertEqual(repn.linear[id(m.v[0])], 0.5)

        repn = visitor.walk_expression(m.int2[1].expr)
        self.assertEqual(len(repn.linear), 2)
        self.assertIn(id(m.v2[1, 1]), repn.linear)
        self.assertEqual(repn.linear[id(m.v2[1, 1])], 0.5)
        self.assertIn(id(m.v2[1, 0]), repn.linear)
        self.assertEqual(repn.linear[id(m.v2[1, 0])], 0.5)

        repn = visitor.walk_expression(m.int4.expr)
        self.assertEqual(len(repn.linear), 4)
        self.assertIn(id(m.v3[1, 10]), repn.linear)
        self.assertEqual(repn.linear[id(m.v3[1, 10])], 1.25)
        self.assertIn(id(m.v3[0, 10]), repn.linear)
        self.assertEqual(repn.linear[id(m.v3[0, 10])], 1.25)
        self.assertIn(id(m.v3[1, 5]), repn.linear)
        self.assertEqual(repn.linear[id(m.v3[1, 5])], 1.25)
        self.assertIn(id(m.v3[0, 5]), repn.linear)
        self.assertEqual(repn.linear[id(m.v3[0, 5])], 1.25)

    # test invalid declarations
    def test_invalid(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 1))
        m.x = ContinuousSet(bounds=(5, 10))
        m.s = Set(initialize=[1, 2, 3])
        m.v = Var(m.t)
        m.v2 = Var(m.s, m.t)
        m.v3 = Var(m.x, m.t)

        def _int(m, t):
            return m.v[t]

        def _int2(m, x, t):
            return m.v3[x, t]

        def _int3(m, s, t):
            return m.v2[s, t]

        # Integrals must be indexed by a ContinuousSet
        with self.assertRaises(ValueError):
            m.int = Integral(rule=_int)

        # Specifying multiple aliases of same option
        with self.assertRaises(TypeError):
            m.int = Integral(m.t, wrt=m.t, withrespectto=m.t, rule=_int)

        # No ContinuousSet specified
        with self.assertRaises(ValueError):
            m.int2 = Integral(m.x, m.t, rule=_int2)

        # 'wrt' is not a ContinuousSet
        with self.assertRaises(ValueError):
            m.int = Integral(m.s, m.t, wrt=m.s, rule=_int2)

        # 'wrt' is not in argument list
        with self.assertRaises(ValueError):
            m.int = Integral(m.t, wrt=m.x, rule=_int)

        # 'bounds' not supported
        with self.assertRaises(DAE_Error):
            m.int = Integral(m.t, wrt=m.t, rule=_int, bounds=(0, 0.5))

        # No rule specified
        with self.assertRaises(ValueError):
            m.int = Integral(m.t, wrt=m.t)

    # test Integral reclassification after discretization
    def test_reclassification_finite_difference(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 1))
        m.x = ContinuousSet(bounds=(5, 10))
        m.s = Set(initialize=[1, 2, 3])
        m.v = Var(m.t)
        m.v2 = Var(m.s, m.t)
        m.v3 = Var(m.t, m.x)

        def _int1(m, t):
            return m.v[t]

        m.int1 = Integral(m.t, rule=_int1)

        def _int2(m, s, t):
            return m.v2[s, t]

        m.int2 = Integral(m.s, m.t, wrt=m.t, rule=_int2)

        def _int3(m, t, x):
            return m.v3[t, x]

        m.int3 = Integral(m.t, m.x, wrt=m.t, rule=_int3)

        def _int4(m, x):
            return m.int3[x]

        m.int4 = Integral(m.x, wrt=m.x, rule=_int4)

        self.assertFalse(m.int1.is_fully_discretized())
        self.assertFalse(m.int2.is_fully_discretized())
        self.assertFalse(m.int3.is_fully_discretized())
        self.assertFalse(m.int4.is_fully_discretized())

        TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.t)

        self.assertTrue(m.int1.is_fully_discretized())
        self.assertTrue(m.int2.is_fully_discretized())
        self.assertFalse(m.int3.is_fully_discretized())
        self.assertFalse(m.int4.is_fully_discretized())

        self.assertTrue(m.int1.ctype is Integral)
        self.assertTrue(m.int2.ctype is Integral)
        self.assertTrue(m.int3.ctype is Integral)
        self.assertTrue(m.int4.ctype is Integral)

        TransformationFactory('dae.finite_difference').apply_to(m, wrt=m.x)

        self.assertTrue(m.int3.is_fully_discretized())
        self.assertTrue(m.int4.is_fully_discretized())

        self.assertTrue(m.int1.ctype is Expression)
        self.assertTrue(m.int2.ctype is Expression)
        self.assertTrue(m.int3.ctype is Expression)
        self.assertTrue(m.int4.ctype is Expression)

    # test Integral reclassification after discretization
    def test_reclassification_collocation(self):
        m = ConcreteModel()
        m.t = ContinuousSet(bounds=(0, 1))
        m.x = ContinuousSet(bounds=(5, 10))
        m.s = Set(initialize=[1, 2, 3])
        m.v = Var(m.t)
        m.v2 = Var(m.s, m.t)
        m.v3 = Var(m.t, m.x)

        def _int1(m, t):
            return m.v[t]

        m.int1 = Integral(m.t, rule=_int1)

        def _int2(m, s, t):
            return m.v2[s, t]

        m.int2 = Integral(m.s, m.t, wrt=m.t, rule=_int2)

        def _int3(m, t, x):
            return m.v3[t, x]

        m.int3 = Integral(m.t, m.x, wrt=m.t, rule=_int3)

        def _int4(m, x):
            return m.int3[x]

        m.int4 = Integral(m.x, wrt=m.x, rule=_int4)

        self.assertFalse(m.int1.is_fully_discretized())
        self.assertFalse(m.int2.is_fully_discretized())
        self.assertFalse(m.int3.is_fully_discretized())
        self.assertFalse(m.int4.is_fully_discretized())

        TransformationFactory('dae.collocation').apply_to(m, wrt=m.t)

        self.assertTrue(m.int1.is_fully_discretized())
        self.assertTrue(m.int2.is_fully_discretized())
        self.assertFalse(m.int3.is_fully_discretized())
        self.assertFalse(m.int4.is_fully_discretized())

        self.assertTrue(m.int1.ctype is Integral)
        self.assertTrue(m.int2.ctype is Integral)
        self.assertTrue(m.int3.ctype is Integral)
        self.assertTrue(m.int4.ctype is Integral)

        TransformationFactory('dae.collocation').apply_to(m, wrt=m.x)

        self.assertTrue(m.int3.is_fully_discretized())
        self.assertTrue(m.int4.is_fully_discretized())

        self.assertTrue(m.int1.ctype is Expression)
        self.assertTrue(m.int2.ctype is Expression)
        self.assertTrue(m.int3.ctype is Expression)
        self.assertTrue(m.int4.ctype is Expression)


if __name__ == "__main__":
    unittest.main()
