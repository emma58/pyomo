# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.common.unittest as unittest

from pyomo.core import ConcreteModel, Var, Param, Constraint, Objective, exp
from pyomo.common.collections import ComponentSet
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import OrderedVarRecorder
from pyomo.core.expr import identify_variables
from pyomo.core.expr.compare import assertExpressionsEqual


class AmplRepnTests(unittest.TestCase):
    def test_divide_by_mutable(self):
        #
        # Test from https://github.com/Pyomo/pyomo/issues/153
        #
        # [ESJ 5/26]: The original issue was with
        # generate_standard_repn and the mutable Param, so
        # this has become a test of the LinearRepnVisitor,
        # which doesn't preserve the Param...
        m = ConcreteModel()
        m.x = Var(bounds=(1, 5))
        m.p = Param(initialize=100, mutable=True)
        m.con = Constraint(expr=exp(5 * (1 / m.x - 1 / m.p)) <= 10)
        m.obj = Objective(expr=m.x**2)

        visitor = LinearRepnVisitor({}, var_recorder=OrderedVarRecorder({}, {}, None))
        test = visitor.walk_expression(m.con.body)
        self.assertEqual(test.constant, 0)
        self.assertEqual(len(test.linear), 0)
        nonlinear_vars = ComponentSet(v for v in identify_variables(test.nonlinear))
        self.assertEqual(len(nonlinear_vars), 1)
        self.assertIn(m.x, nonlinear_vars)
        assertExpressionsEqual(self, test.nonlinear, exp((1 / m.x - 0.01) * 5))


if __name__ == "__main__":
    unittest.main()
