#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base import (
    TransformationFactory, Reals, Objective, Constraint, Var, Block, minimize,
    maximize, NonPositiveReals, NonNegativeReals, Reals, SortComponents, Any,
    value)
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.core.expr.numvalue import is_potentially_variable
from pyomo.common.modeling import unique_component_name
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.repn import generate_standard_repn
from pyomo.contrib.fme.fourier_motzkin_elimination import \
    vars_to_eliminate_list as var_list
# register the transformation
from pyomo.contrib.convex_programming_duality.\
    move_var_bounds_to_constraints import Move_Variable_Bounds_to_Constraints

from pytest import set_trace

@TransformationFactory.register('contrib.linear_programming_dual',
                                doc="Take the dual of linear programs")
class Linear_Programming_Dual(Transformation):
    """Take dual of linear programs. That is, given problems of the form:
    min  c^Tx
    s.t. Ax >= b
          x >= 0

    this transformation will deactivate the original model and add a block
    containing the model:

    max  b^Ty
    s.t. A^Ty <= c
         y >= 0
    """
    CONFIG = ConfigBlock("contrib.linear_programming_dual")
    CONFIG.declare('assume_fixed_vars_permanent', ConfigValue(
        default=False,
        domain=bool,
        description="Whether or not to transform assuming that "
        "fixed variables could eventually be unfixed. Set to "
        "True if fixed variables should be moved to the right-hand "
        "side and treated as data.",
        doc="""
        Whether or not to treat fixed variables as data or variables.

        When True, fixed variables will be moved to the right-hand
        side and treated as data. This means that if they are later
        unfixed, the dual will no longer be correct.

        When False, fixed variables will be treated as primal
        variables, and the dual will be correct whether or not they
        are later unfixed.
        """
    ))
    CONFIG.declare('dual_block_name', ConfigValue(
        default=None,
        domain=str,
        description="Optional name for the Block that will hold the "
        "dual model.",
        doc="""
        Optional name (str) to use for the Block that the
        transformation creates to hold the dual model. Note that
        this name must be unique with respect to the Block the
        transformation is applied to.

        If not specified, the transformation will generate a unique
        name for the Block beginning with:
        '_pyomo_contrib_linear_programming_dual'
        """
    ))
    CONFIG.declare('treat_as_data', ConfigValue(
        default=[],
        domain=var_list,
        description="List of primal variables that should be treated as data. "
        "That is, the primal does not optimize them, so the dual should be "
        "taken treating them as coefficients or right-hand side constants.",
        doc="""
        List of primal variables that should be treated as data. (This is a
        common situation in bilevel programming, when taking the dual of an
        inner problem that involves but does not optimize outer-problem
        variables.) Variables in this list will be treated as coefficients or
        right-hand-side constants, depending on where they appear.
        """
    ))
    _domains = {
        (minimize, 'leq') : NonPositiveReals,
        (minimize, 'geq') : NonNegativeReals,
        (maximize, 'leq') : NonNegativeReals,
        (maximize, 'geq') : NonPositiveReals
    }

    def __init__(self):
        super(Linear_Programming_Dual, self).__init__()

    def _apply_to(self, instance, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        try:
            fixed_vars = []
            if not config.assume_fixed_vars_permanent:
                for v in get_vars_from_components(instance, ctype=(Constraint,
                                                                   Objective),
                                                  include_fixed=True,
                                                  active=True,
                                                  descend_into=Block):
                    if v.fixed:
                        fixed_vars.append(v)
                        v.unfix()
            for v in config.treat_as_data:
                v.fix()

            TransformationFactory(
                'contrib.move_var_bounds_to_constraints').apply_to(instance)

            primal_obj = self._get_objective(instance)
            primal_obj.deactivate()
            obj_coefs = self._get_obj_coef_map(primal_obj.expr)

            dual_block = self._create_transformation_block(
                instance, config.dual_block_name)

            (primal_constraints,
             primal_variables) = self._create_dual_variables(instance,
                                                             dual_block,
                                                             primal_obj.sense)

            self._create_dual_objective_and_constraints(primal_constraints,
                                                        primal_variables,
                                                        obj_coefs, dual_block,
                                                        primal_obj.sense)
        finally:
            for v in config.treat_as_data:
                v.unfix()
            for v in fixed_vars:
                v.fix()

    def _get_objective(self, instance):
        active_objectives = [obj for obj in
                             instance.component_data_objects(
                                 Objective, active=True, descend_into=Block)]
        if len(active_objectives) > 1:
            raise ValueError("There is more than one active objective on "
                             "the primal model: %s" %
                             ", ".join([obj.name for obj in active_objectives]))
        return active_objectives[0]

    def _get_obj_coef_map(self, expr):
        repn = generate_standard_repn(expr, compute_values=False)
        if not repn.is_linear():
            raise ValueError("The active objective is nonlinear!")
        return ComponentMap([(var, coef) for coef, var in
                             zip(repn.linear_coefs, repn.linear_vars)])

    def _create_transformation_block(self, instance, nm):
        if nm is None:
            nm = unique_component_name(
                instance, '_pyomo_contrib_linear_programming_dual')
        transBlock = Block()
        instance.add_component(nm, transBlock)

        return transBlock

    def _create_dual_variables(self, instance, dual_block, sense):
        primal_constraints = ComponentMap()
        primal_vars = ComponentSet()
        primal_var_list = []
        for cons in instance.component_objects(
                Constraint, active=True, descend_into=Block,
                sort=SortComponents.deterministic):
            if cons.is_indexed():
                duals = Var(Any, dense=False)
            else:
                duals = Var()
            dual_block.add_component(unique_component_name(
                dual_block, '%s_dual' % cons.name), duals)
            for idx, cons in cons.items():
                cons.deactivate()
                # idx could be None if this thing wasn't indexed, but it's okay
                lower = cons.lower
                upper = cons.upper
                body = generate_standard_repn(cons.body, compute_values=False)
                if not body.is_linear():
                    # Remind people to read:
                    raise ValueError(
                        "Detected nonlinear constraint body in constraint "
                        "'%s': The 'linear_programming_dual' transformation "
                        "requires all the constraints be linear." % cons.name)
                coef_map = ComponentMap()
                for coef, var in zip(body.linear_coefs, body.linear_vars):
                    if var not in primal_vars:
                        primal_vars.add(var)
                        primal_var_list.append(var)
                    coef_map[var] = coef
                body_constant = body.constant

                if lower is not None and upper is not None:
                    # It's an equality
                    if value(lower) == value(upper):
                        duals[idx].domain = Reals
                        primal_constraints[duals[idx]] = ('eq', coef_map,
                                                          lower - body_constant)
                    else: # It's actually two constraints: a <= and a >=
                        leq_idx = '%s_leq' % idx
                        duals[leq_idx].domain = self._domains[sense, 'leq']
                        primal_constraints[duals[leq_idx]] = ('leq', coef_map,
                                                              upper -
                                                              body_constant)

                        geq_idx = '%s_geq' % idx
                        duals[geq_idx].domain = self._domains[sense, 'geq']
                        primal_constraints[duals[geq_idx]] = ('geq', coef_map,
                                                              lower -
                                                              body_constant)
                # >=
                elif lower is not None:
                    duals[idx].domain = self._domains[sense, 'geq']
                    primal_constraints[duals[idx]] = ('geq', coef_map, lower -
                                                      body_constant)
                # <=
                elif upper is not None:
                    duals[idx].domain = self._domains[sense, 'leq']
                    primal_constraints[duals[idx]] = ('leq', coef_map, upper -
                                                      body_constant)

        return (primal_constraints, primal_var_list)

    def _create_dual_objective_and_constraints(self, primal_constraints,
                                               primal_variables, obj_coefs,
                                               dual_block, sense):
        dual_constraint_exprs = {}
        constraints_by_container = ComponentMap()
        for primal_var in primal_variables:
            idx = primal_var.index()
            parent = primal_var.parent_component()
            if parent.is_indexed():
                if parent not in constraints_by_container.keys():
                    dual_cons = Constraint(parent.index_set())
                    dual_block.add_component(unique_component_name(dual_block,
                                                                   parent.name),
                                             dual_cons)
                    # This is just a dummy expression so that the constraint
                    # will exist...
                    dual_cons[idx] = primal_var == 0
                    dual_consData = dual_cons[idx]
                    constraints_by_container[parent] = dual_cons
                else:
                    # This is just a dummy expression so that the constraint
                    # will exist...
                    constraints_by_container[parent][idx] = primal_var == 0
                    dual_consData = constraints_by_container[parent][idx]
            else:
                dual_consData = Constraint()
                dual_block.add_component(unique_component_name(dual_block,
                                                               primal_var.name),
                                         dual_consData)
            if primal_var.domain is Reals:
                dual_cons_type = 'eq'
            elif primal_var.domain is NonNegativeReals:
                dual_cons_type = 'leq' if sense == minimize else 'geq'
            elif primal_var.domain is NonPositiveReals:
                dual_cons_type = 'geq' if sense == minimize else 'leq'
            dual_rhs = obj_coefs.get(primal_var, 0)
            # build the expression
            expr = 0
            for dual_var, (cons_type, coef_map, primal_rhs) in \
                primal_constraints.items():
                coef = coef_map.get(primal_var, 0)
                if is_potentially_variable(coef) or value(coef) != 0:
                    expr += coef*dual_var
            dual_constraint_exprs[dual_consData] = [dual_cons_type, expr,
                                                    dual_rhs]

        # now set all the expressions:
        for cons, (cons_type, expr, rhs) in dual_constraint_exprs.items():
            if cons_type == 'eq':
                cons.set_value(expr == rhs)
            elif cons_type == 'leq':
                cons.set_value(expr <= rhs)
            else:
                cons.set_value(expr >= rhs)

        obj_expr = 0
        for dual_var, (cons_type, coef_map, rhs) in \
                primal_constraints.items():
            if is_potentially_variable(rhs) or value(rhs) != 0:
                obj_expr += rhs*dual_var
        dual_block.add_component(unique_component_name(dual_block, "dual_obj"),
                                 Objective(expr=obj_expr, sense=minimize if
                                           sense == maximize else maximize))
