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

from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.contrib.gdpopt.config_options import (
    _add_mip_solver_configs, _add_nlp_solver_configs, _add_tolerance_configs,
    _add_OA_configs)
from pyomo.contrib.gdpopt.create_oa_subproblems import (
    _get_master_and_subproblem)
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.oa_algorithm_utils import (
    _fix_master_soln_solve_subproblem_and_add_cuts)
from pyomo.contrib.gdpopt.solve_master_problem import solve_MILP_master_problem
from pyomo.contrib.gdpopt.util import time_code
from pyomo.core import Objective

# ESJ: In the future, if we have a direct interface to cplex or gurobi, we
# should get the integer solutions several-at-a-time with a solution pool or
# something of the like...

class _GDP_RIC_Solver():
    """The GDPopt (Generalized Disjunctive Programming optimizer) relaxation
    with integer cuts (RIC) solver.

    Accepts models that can include nonlinear, continuous variables and
    constraints, as well as logical conditions. For non-convex problems, RIC
    will not be exact unless the NLP subproblems are solved globally.
    """
    CONFIG = ConfigBlock("GDPoptRIC")
    _add_mip_solver_configs(CONFIG)
    _add_nlp_solver_configs(CONFIG)
    _add_tolerance_configs(CONFIG)
    _add_OA_configs(CONFIG)

    def __init__(self, parent):
        self.parent = parent
        # Transfer the parent config info: we create it if it is not there, and
        # overwrite the values if it is already there. The parent defers to what
        # is in this class during solve.
        for kwd, val in self.parent.CONFIG.items():
            if kwd not in self.CONFIG:
                self.CONFIG.declare(kwd, ConfigValue(default=val))
            else:
                self.CONFIG[kwd] = val

    def _solve_gdp(self, original_model, config):
        logger = config.logger

        (master_util_block,
         subproblem_util_block) = _get_master_and_subproblem(self, config)
        master = master_util_block.model()
        subproblem = subproblem_util_block.model()
        master_obj = next(master.component_data_objects(Objective, active=True,
                                                        descend_into=True))

        self.parent._log_header(logger)

        # main loop
        while self.parent.iteration < config.iterlim:
            self.parent.iteration += 1

            # solve linear master problem
            with time_code(self.parent.timing, 'mip'):
                mip_feasible = solve_MILP_master_problem(master_util_block,
                                                         config,
                                                         self.parent.timing)
                self.parent._update_bounds_after_master_problem_solve(
                    mip_feasible, master_obj, logger)

            # Check termination conditions
            if self.parent.any_termination_criterion_met(config):
                break

            with time_code(self.parent.timing, 'nlp'):
                _fix_master_soln_solve_subproblem_and_add_cuts(
                    master_util_block, subproblem_util_block, config, self)

            # Add integer cut
            with time_code(self.parent.timing, "integer cut generation"):
                add_no_good_cut(master_util_block, config)

            # Check termination conditions
            if self.parent.any_termination_criterion_met(config):
                break

    def _add_cuts_to_master_problem(self, subproblem_util_block,
                                    master_util_block, objective_sense, config,
                                    timing):
        # Nothing to do here
        pass
