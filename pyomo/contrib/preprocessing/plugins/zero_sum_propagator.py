# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

"""Transformation to propagate a zero value to terms of a sum."""

from pyomo.core.base.transformation import TransformationFactory
from pyomo.core.base.constraint import Constraint
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import OrderedVarRecorder


@TransformationFactory.register(
    'contrib.propagate_zero_sum',
    doc="Propagate fixed-to-zero for sums of only positive (or negative) vars.",
)
class ZeroSumPropagator(IsomorphicTransformation):
    """Propagates fixed-to-zero for sums of only positive (or negative) vars.

    If :math:`z` is fixed to zero and :math:`z = x_1 + x_2 + x_3` and
    :math:`x_1`, :math:`x_2`, :math:`x_3` are all non-negative or all
    non-positive, then :math:`x_1`, :math:`x_2`, and :math:`x_3` will be fixed
    to zero.

    """

    def _apply_to(self, instance):
        visitor = LinearRepnVisitor({}, var_recorder=OrderedVarRecorder({}, {}, None))
        for constr in instance.component_data_objects(
            ctype=Constraint, active=True, descend_into=True
        ):
            if not constr.body.polynomial_degree() == 1:
                continue  # constraint not linear. Skip.

            repn = visitor.walk_expression(constr.body)
            if constr.has_ub() and repn.constant == value(constr.upper):
                # term1 + term2 + term3 + ... <= 0
                # all var terms need to be non-negative
                if all(
                    # variable is non-negative and has non-negative coefficient
                    (
                        visitor.var_map[vid].has_lb()
                        and value(visitor.var_map[vid].lb) >= 0
                        and coef >= 0
                    )
                    or
                    # variable is non-positive and has non-positive coefficient
                    (
                        visitor.var_map[vid].has_ub()
                        and value(visitor.var_map[vid].ub) <= 0
                        and coef <= 0
                    )
                    for vid, coef in repn.linear.items()
                ):
                    for vid in repn.linear:
                        visitor.var_map[vid].fix(0)
                    continue
            if constr.has_lb() and repn.constant == value(constr.lower):
                # term1 + term2 + term3 + ... >= 0
                # all var terms need to be non-positive
                if all(
                    # variable is non-negative and has non-positive coefficient
                    (
                        visitor.var_map[vid].has_lb()
                        and value(visitor.var_map[vid].lb) >= 0
                        and coef <= 0
                    )
                    or
                    # variable is non-positive and has non-negative coefficient
                    (
                        visitor.var_map[vid].has_ub()
                        and value(visitor.var_map[vid].ub) <= 0
                        and coef >= 0
                    )
                    for vid, coef in repn.linear.items()
                ):
                    for vid in repn.linear:
                        visitor.var_map[vid].fix(0)
