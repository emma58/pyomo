# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

# -*- coding: UTF-8 -*-
"""Transformation to remove zero terms from constraints."""

from pyomo.core import quicksum
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.transformation import TransformationFactory
import pyomo.core.expr as EXPR
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.util import OrderedVarRecorder
from pyomo.common.config import ConfigDict, ConfigValue


@TransformationFactory.register(
    'contrib.remove_zero_terms', doc="Remove terms 0 * var in constraints"
)
class RemoveZeroTerms(IsomorphicTransformation):
    """Looks for :math:`0 v` in a constraint and removes it.

    Currently limited to processing linear constraints of the form :math:`x_1 =
    0 x_3`, occurring as a result of fixing :math:`x_2 = 0`.

    .. note:: TODO: support nonlinear expressions

    """

    CONFIG = ConfigDict("RemoveZeroTerms")
    CONFIG.declare(
        "constraints_modified",
        ConfigValue(
            default={},
            description="A dictionary that maps the constraints modified during "
            "the transformation to a tuple: (original_expr, modified_expr)",
        ),
    )

    def _apply_to(self, model, **kwargs):
        """Apply the transformation."""
        config = self.CONFIG(kwargs)
        m = model
        visitor = LinearRepnVisitor({}, var_recorder=OrderedVarRecorder({}, {}, None))

        for constr in m.component_data_objects(
            ctype=Constraint, active=True, descend_into=True
        ):
            repn = visitor.walk_expression(constr.body)
            # Check repn is nonlinear and non-constant
            if repn.nonlinear is not None or not repn.linear:
                continue  # we currently only process linear constraints, and we
                # assume that trivial constraints have already been
                # deactivated or will be deactivated in a different
                # step

            original_expr = constr.expr
            const = repn.constant

            # reconstitute the constraint, keeping only nonzero-coefficient terms
            # (LinearRepnVisitor already filters zero coefficients in finalizeResult)
            constr_body = (
                quicksum(coef * visitor.var_map[vid] for vid, coef in repn.linear.items())
                + const
            )
            if constr.equality:
                new_expr = constr_body == constr.upper
            elif constr.has_lb() and not constr.has_ub():
                new_expr = constr_body >= constr.lower
            elif constr.has_ub() and not constr.has_lb():
                new_expr = constr_body <= constr.upper
            else:
                # constraint is a bounded inequality of form a <= x <= b.
                new_expr = EXPR.inequality(constr.lower, constr_body, constr.upper)
            constr.set_value(new_expr)
            config.constraints_modified[constr] = (original_expr, new_expr)
