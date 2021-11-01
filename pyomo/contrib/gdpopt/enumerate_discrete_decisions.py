#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.contrib.gdpopt.nlp_solve import solve_global_subproblem
from pyomo.contrib.gdpopt.data_class import MasterProblemResult
from pyomo.contrib.gdpopt.iterate import _terminate_at_iteration_limit

from pyomo.opt import TerminationCondition as tc
from pyomo.common.collections import ComponentSet

from itertools import product

def _enumerate_discrete_decisions(solve_data, config):
    """Main loop for the enumeration strategy. Note that we enumerate over all 
    the possible decisions for the indicator variables, so the subproblems 
    might be MINLPs if there are discrete decisions on the model.

    Returns True if it enumerates all decisions or terminates at the iteration 
    limit.
    """
    # Build out the list of BooleanVars for each of the Disjuncts, grouped by
    # Disjunction. We will then just enumerate over the cartesian product of
    # this list of sets, and that will give the set of BooleanVars to set to
    # True
    m = solve_data.working_model
    GDPopt_util = m.GDPopt_utils
    disjunctions = []
    for i, disjunction in enumerate(GDPopt_util.disjunction_list):
        disjunctions.append([])
        disjuncts = disjunctions[i]
        for disjunct in disjunction.disjuncts:
            v = disjunct.indicator_var
            disjuncts.append(v)
            v.fix(False)

    for true_vars in product(*disjunctions):
        solve_data.master_iteration += 1
        # print line for visual display
        config.logger.info( '---GDPopt Iteration %s---' %
                            solve_data.master_iteration)

        for v in true_vars:
            v.fix(True)
        # all the other vars are already False

        mip_results = MasterProblemResult()
        mip_results.disjunct_values = list(disj.binary_indicator_var.value for
                                           disj in GDPopt_util.disjunct_list)

        # solve the NLP subproblem globally. (This will keep track of the best
        # solution found)
        nlp_result = solve_global_subproblem(mip_results, solve_data, config)

        if solve_data.master_iteration >= config.iterlim:
            _terminate_at_iteration_limit(solve_data, config)
            return True

        # restore state of boolean vars
        for v in true_vars:
            v.fix(False)

    # By virtue of finishing the above loop, the bounds have closed.
    solve_data.LB = solve_data.UB
    solve_data.results.solver.termination_condition = tc.optimal
    
    return True
