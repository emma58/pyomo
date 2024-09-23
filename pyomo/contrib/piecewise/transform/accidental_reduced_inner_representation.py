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

from pyomo.contrib.piecewise.transform.reduced_inner_representation_gdp import (
    ReducedInnerRepresentationGDPTransformation,
)
from pyomo.core import Constraint
from pyomo.core.base import TransformationFactory


@TransformationFactory.register(
    'contrib.piecewise.accidental_reduced_inner_repn_gdp',
    doc="Convert piecewise-linear model to a GDP "
    "using an inner representation of the "
    "simplices that are the domains of the linear "
    "functions.",
)
class AccidentalReducedInnerRepresentationGDPTransformation(
    ReducedInnerRepresentationGDPTransformation
):
    """ """

    CONFIG = ReducedInnerRepresentationGDPTransformation.CONFIG()
    _transformation_name = 'pw_linear_accidental_reduced_inner_repn'

    def _add_disjunctive_constraints(
        self, disj, transBlock, extreme_pts, num_extreme_pts
    ):
        disj.lambdas_sum_to_one = Constraint(
            expr=sum(transBlock.lambdas[i] for i in extreme_pts) >= 1
        )
