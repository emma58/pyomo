#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core.base import TransformationFactory
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.plugins.transform.hierarchy import Transformation

@TransformationFactory.register('contrib.convex_linearly_constrained_dual',
                                doc="Take the dual of linear programs and "
                                "convex linearly-constrained quadratic "
                                "programs.")
class Linearly_Constrained_Dual(Transformation):
    """Take dual of linear programs and of convex linearly-constrained quadratic
    programs.

    That is, given problems of the form:
    min  c^Tx
    s.t. Ax >= b
          x >= 0

    or:

    min  (1/2)x^TDx + c^Tx
    s.t. Ax >= b,

    this transformation will deactivate the original model and add a block 
    containing the model:

    max  b^Ty
    s.t. A^Ty <= c
         y >= 0

    or 

    max  -(1/2)t^TDt + b^ty
    s.t. A^Ty + Dt = c
         y >= 0

    respectively.
    """
    CONFIG = ConfigBlock("contrib.linearly_constrained_dual")
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
        '_pyomo_contrib_linearly_constrained_dual'
        """
    ))

    def __init__(self):
        super(Linearly_Constrained_Dual, self).__init__()

    def _apply_to(self, instance, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        TransformationFactory(
            'contrib.move_var_bounds_to_constraints').apply_to(instance)

        (primal_obj, 
         primal_constraints, 
         primal_variables) = self._gather_primal_problem(instance)

    def _gather_primal_problem(self, instance):
        active_objectives = [obj for obj in
                             instance.component_data_objects(
                                 Objective, active=True, descend_into=Block)]
        if len(active_objectives) > 1:
            raise ValueError("There is more than one active objective on "
                             "the primal model: %s" % 
                             ", ".join([obj.name for obj in active_objectives]))

        
