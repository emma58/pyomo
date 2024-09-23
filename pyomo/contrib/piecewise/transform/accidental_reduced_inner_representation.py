#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.piecewise.transform.piecewise_linear_transformation_base import (
    PiecewiseLinearTransformationBase,
)
from pyomo.core import Constraint, NonNegativeIntegers, Var
from pyomo.core.base import TransformationFactory
from pyomo.gdp import Disjunct, Disjunction


@TransformationFactory.register(
    'contrib.piecewise.accidental_reduced_inner_repn_gdp',
    doc="Convert piecewise-linear model to a GDP "
    "using an inner representation of the "
    "simplices that are the domains of the linear "
    "functions.",
)
class AccidentalReducedInnerRepresentationGDPTransformation(
        PiecewiseLinearTransformationBase):
    """
    
    """
    CONFIG = PiecewiseLinearTransformationBase.CONFIG()
    _transformation_name = 'pw_linear_accidental_reduced_inner_repn'

    def _transform_pw_linear_expr(self, pw_expr, pw_linear_func, transformation_block):
        transBlock = transformation_block.transformed_functions[
            len(transformation_block.transformed_functions)
        ]

        # get the PiecewiseLinearFunctionExpression
        dimension = pw_expr.nargs()
        transBlock.disjuncts = Disjunct(NonNegativeIntegers)
        substitute_var = transBlock.substitute_var = Var()
        pw_linear_func.map_transformation_var(pw_expr, substitute_var)
        substitute_var_lb = float('inf')
        substitute_var_ub = -float('inf')
        extreme_pts_by_simplex = {}
        linear_func_by_extreme_pt = {}
        # Save all the extreme points as sets since we will need to check set
        # containment to build the constraints fixing the multipliers to 0. We
        # can also build the data structure that will allow us to later build
        # the linear func expression
        for simplex, linear_func in zip(
            pw_linear_func._simplices, pw_linear_func._linear_functions
        ):
            extreme_pts = extreme_pts_by_simplex[simplex] = set()
            for idx in simplex:
                extreme_pts.add(idx)
                if idx not in linear_func_by_extreme_pt:
                    linear_func_by_extreme_pt[idx] = linear_func

            # We're going to want bounds on the substitute var, so we use
            # interval arithmetic to figure those out as we go.
            (lb, ub) = compute_bounds_on_expr(linear_func(*pw_expr.args))
            if lb is not None and lb < substitute_var_lb:
                substitute_var_lb = lb
            if ub is not None and ub > substitute_var_ub:
                substitute_var_ub = ub

        # set the bounds on the substitute var
        if substitute_var_lb < float('inf'):
            transBlock.substitute_var.setlb(substitute_var_lb)
        if substitute_var_ub > -float('inf'):
            transBlock.substitute_var.setub(substitute_var_ub)

        num_extreme_pts = len(pw_linear_func._points)
        # lambda[i] will be the multiplier for the extreme point with index i in
        # pw_linear_fun._points
        transBlock.lambdas = Var(range(num_extreme_pts), bounds=(0, 1))

        # Now that we have all of the extreme points, we can make the
        # disjunctive constraints
        for simplex in pw_linear_func._simplices:
            disj = transBlock.disjuncts[len(transBlock.disjuncts)]
            # ESJ TODO: Note that this next constraint is the only change from
            # the reduced inner formulation, so if this works, I should
            # centralize the rest.
            disj.lambdas_sum_to_one = Constraint(
                expr=sum(transBlock.lambdas[i] for i in
                         extreme_pts_by_simplex[simplex]) >= 1
            )

        # Make the disjunction
        transBlock.pick_a_piece = Disjunction(
            expr=[d for d in transBlock.disjuncts.values()]
        )

        # Now we make the global constraints
        transBlock.convex_combo = Constraint(
            expr=sum(transBlock.lambdas[i] for i in range(num_extreme_pts)) == 1
        )
        transBlock.linear_func = Constraint(
            expr=sum(
                linear_func_by_extreme_pt[j](*pt) * transBlock.lambdas[j]
                for (j, pt) in enumerate(pw_linear_func._points)
            )
            == substitute_var
        )

        @transBlock.Constraint(range(dimension))
        def linear_combo(b, i):
            return pw_expr.args[i] == sum(
                pt[i] * transBlock.lambdas[j]
                for (j, pt) in enumerate(pw_linear_func._points)
            )

        return transBlock.substitute_var
