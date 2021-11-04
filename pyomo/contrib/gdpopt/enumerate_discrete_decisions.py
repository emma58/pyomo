#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Utility functions for the enumeration strategy."""

from itertools import product
from nose.tools import set_trace
from pyomo.common.collections import ComponentSet

def _precalculate_discrete_solutions(disjunctions, non_indicator_boolean_vars,
                                     discrete_var_values):
    # now we will calculate all the possible indicator_var realizations, and
    # then multiply those out by all the boolean var realizations and all the
    # integer var realizations.
    realizations = []
    for true_indicators in product(*disjunctions):
        for boolean_realization in product(
                [True, False], repeat=len(non_indicator_boolean_vars)):
            for integer_realization in product(*discrete_var_values):
                yield (ComponentSet(true_indicators), boolean_realization,
                       integer_realization)

def _fix_discrete_solution(solve_data, idx):
    util_blk = solve_data.linear_GDP.GDPopt_utils
    (true_indicators, boolean_realization,
     integer_realization) = util_blk.discrete_realizations[idx]

    for boolean_var in util_blk.boolean_indicator_vars:
        binary = 1 if boolean_var in true_indicators else 0
        # 'Fix' to value, but do it via the bounds so that we will have
        # something sent to solver, making *it* check the feasibility problem.
        boolean_var.get_associated_binary().setlb(binary)
        boolean_var.get_associated_binary().setub(binary)

    for var, val in zip(util_blk.non_indicator_boolean_vars,
                             boolean_realization):
        binary = 1 if val else 0
        var.get_associated_binary().setlb(binary)
        var.get_associated_binary().setub(binary)

    for var, val in zip(util_blk.non_indicator_discrete_vars,
                             integer_realization):
        var.fix(val)
