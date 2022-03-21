#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base import (TransformationFactory, Block, Constraint, Var,
                             SortComponents, is_potentially_variable)
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.common.modeling import unique_component_name

from pytest import set_trace

@TransformationFactory.register('contrib.move_var_bounds_to_constraints',
                                doc="Release bounds on Var objects and add "
                                "them as constraints on the model.")
class Move_Variable_Bounds_to_Constraints(Transformation):
    """
    Set bounds on Var objects to 'None', adding existing bounds onto the model
    as constraints. Does not effect the domain of the variables.
    """
    CONFIG = ConfigBlock("contrib.move_var_bounds_to_constraints")
    CONFIG.declare('bound_constraint_block_name', ConfigValue(
        default=None,
        domain=str,
        description="Optional name for the Block that will hold the "
        "new constraints.",
        doc="""
        Optional name (str) to use for the Block that the
        transformation creates to hold the constraints enforcing
        the variable bounds.

        If not specified, the transformation will generate a unique
        name for the Block beginning with:
        '_pyomo_contrib_var_bounds_constraints'
        """
    ))
    def __init__(self):
        super(Move_Variable_Bounds_to_Constraints, self).__init__()

    def _apply_to(self, instance, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        # create block to put new constraints on
        constraint_block = Block()
        if config.bound_constraint_block_name is not None:
            instance.add_component(config.bound_constraint_block_name,
                                   constraint_block)
        else:
            instance.add_component(unique_component_name(
                instance,
                '_pyomo_contrib_var_bounds_constraints'), constraint_block)

        # I'm going to actually use component_data_objects here so that we
        # transform even variables that aren't used elsewhere on the model. I
        # think this is safer because this way if you add to the model later
        # thinking you've moved all your bounds, you still have (as long as you
        # don't add variables with bounds, but that's your own fault).
        for v in instance.component_data_objects(
                Var, descend_into=Block, sort=SortComponents.deterministic):
            self._move_var_bounds_to_constraint(v, constraint_block)

    def _move_var_bounds_to_constraint(self, var, constraint_block):
        # Get bound expressions, not values:
        lower = var.lower
        upper = var.upper
        if lower is not None:
            # Only add a constraint if lower is potentially variable or if it's
            # not and the bound is not redundant with the domain.
            dlb = var.domain.bounds()[0]
            if not (not is_potentially_variable(lower) and dlb is not None and
                    var.lb <= dlb):
                lower_cons = Constraint(expr=var >= lower)
                constraint_block.add_component(unique_component_name(
                    constraint_block, '%s_lb' % var.name), lower_cons)
            var.setlb(None)
        if upper is not None:
            dub = var.domain.bounds()[1]
            if not (not is_potentially_variable(upper) and dub is not None and
                    var.ub >= dub):
                upper_cons = Constraint(expr=var <= upper)
                constraint_block.add_component(unique_component_name(
                    constraint_block, '%s_ub' % var.name), upper_cons)
            var.setub(None)
