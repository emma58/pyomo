#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

from pyomo.common.modeling import unique_component_name
from pyomo.core import (
    TransformationFactory, Transformation, Block, VarList, Set,
    SortComponents, Objective, Constraint, ConstraintList, Any)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn

logger = logging.getLogger('pyomo.gdp')


@TransformationFactory.register(
    'gdp.bilinear',
    doc="Creates a disjunctive model where bilinear terms are replaced with "
    "disjunctive expressions.")
class Bilinear_Transformation(Transformation):
    """
    TODO: 
    * write a docstring
    * allow targets (so that you can do this inside Disjuncts too)
    """
    def __init__(self):
        super(Bilinear_Transformation, self).__init__()
        self._transformation_blocks = {}

    def _apply_to(self, instance, **kwds):
        try:
            self._apply_to_impl(instance, **kwds)
        finally:
            self._transformation_blocks.clear()

    def _apply_to_impl(self, instance, **kwds):
        transBlock = self._add_transformation_block(instance)
        #
        # Iterate over all blocks
        #
        for block in instance.block_data_objects(
                active=True, sort=SortComponents.deterministic):
            self._transformBlock(block, transBlock)

    def _add_transformation_block(self, to_block):
        if to_block in self._transformation_blocks:
            return self._transformation_blocks[to_block]

        # make a transformation block on to_block to put transformed disjuncts
        # on
        transBlockName = unique_component_name(
            to_block,
            '_pyomo_gdp_bilinear_reformulation')
        self._transformation_blocks[to_block] = transBlock = Block()
        to_block.add_component(transBlockName, transBlock)
        transBlock.cache = {}
        transBlock.vlist = VarList()
        transBlock.disjuncts_ = Disjunct(Any)
        transBlock.disjunctions = Disjunction(Any)
        transBlock.constraints = ConstraintList()

        return transBlock

    def _transformBlock(self, block, transBlock):
        for component in block.component_objects(Objective, active=True,
                                                 descend_into=False):
            expr = self._transformExpression(component.expr, transBlock)
            component.expr = expr
        for component in block.component_data_objects(Constraint, active=True,
                                                      descend_into=False):
            expr = self._transformExpression(component.body, transBlock)
            component._body = expr

    def _transformExpression(self, expr, transBlock):
        if expr.polynomial_degree() > 2:
            raise ValueError(
                "Cannot transform polynomial terms with degree > 2")
        if expr.polynomial_degree() < 2:
            return expr
        #
        expr = self._replace_bilinear(expr, transBlock)
        return expr

    def _replace_bilinear(self, expr, transBlock):
        idMap = {}
        terms = generate_standard_repn(expr, idMap=idMap)
        # Constant
        e = terms.constant
        # Linear terms
        for var, coef in zip(terms.linear_vars, terms.linear_coefs):
            e += coef * var
        # Quadratic terms
        if len(terms.quadratic_coefs) > 0:
            for vars_, coef_ in zip(terms.quadratic_vars,
                                    terms.quadratic_coefs):
                #
                if vars_[0].is_binary():
                    v = transBlock.cache.get((id(vars_[0]), id(vars_[1])), None)
                    if v is None:
                        v = transBlock.vlist.add()
                        transBlock.cache[id(vars_[0]), id(vars_[1])] = v
                        bounds = vars_[1].bounds
                        v.setlb(bounds[0])
                        v.setub(bounds[1])
                        id_ = len(transBlock.vlist)
                        # First disjunct
                        d0 = transBlock.disjuncts_[id_,0]
                        #d0.c1 = Constraint(expr=vars_[0] == 1)
                        d0.c2 = Constraint(expr=v == vars_[1])
                        # Second disjunct
                        d1 = transBlock.disjuncts_[id_,1]
                        #d1.c1 = Constraint(expr=vars_[0] == 0)
                        d1.c2 = Constraint(expr=v == 0)
                        # Disjunction
                        transBlock.disjunctions[id_] = [
                            transBlock.disjuncts_[id_,0],
                            transBlock.disjuncts_[id_,1]
                        ]
                        # link the original binary and the indicator var
                        transBlock.constraints.add(
                            vars_[0] == d0.binary_indicator_var)
                    # The disjunctive variable is the expression
                    e += coef_*v
                #
                elif vars_[1].is_binary():
                    v = transBlock.cache.get((id(vars_[1]), id(vars_[0])), None)
                    if v is None:
                        v = transBlock.vlist.add()
                        transBlock.cache[id(vars_[1]), id(vars_[0])] = v
                        bounds = vars_[0].bounds
                        v.setlb(bounds[0])
                        v.setub(bounds[1])
                        id_ = len(transBlock.vlist)
                        # First disjunct
                        d0 = transBlock.disjuncts_[id_,0]
                        d0.c1 = Constraint(expr=vars_[1] == 1)
                        d0.c2 = Constraint(expr=v == vars_[0])
                        # Second disjunct
                        d1 = transBlock.disjuncts_[id_,1]
                        d1.c1 = Constraint(expr=vars_[1] == 0)
                        d1.c2 = Constraint(expr=v == 0)
                        # Disjunction
                        transBlock.disjunctions[id_] = [
                            transBlock.disjuncts_[id_,0],
                            transBlock.disjuncts_[id_,1]]
                    # The disjunctive variable is the expression
                    e += coef_*v
                else:
                    # If neither variable is boolean, just reinsert the original
                    # bilinear term
                    e += coef_*vars_[0]*vars_[1]
        return e
