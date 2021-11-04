"""Iteration code."""
from __future__ import division

from pyomo.contrib.gdpopt.cut_generation import (add_integer_cut,
                                                 add_outer_approximation_cuts,
                                                 add_affine_cuts)
from pyomo.contrib.gdpopt.mip_solve import solve_LOA_master
from pyomo.contrib.gdpopt.nlp_solve import (solve_global_subproblem,
                                            solve_local_subproblem)
from pyomo.opt import TerminationCondition as tc
from pyomo.core.base.objective import Objective, minimize
from pyomo.core.expr.numvalue import value
from pyomo.contrib.gdpopt.util import time_code, get_main_elapsed_time
from pyomo.contrib.gdpopt.enumerate_discrete_decisions import (
    _fix_discrete_solution)
from nose.tools import set_trace

def GDPopt_iteration_loop(solve_data, config):
    """Algorithm main loop.

    Returns True if successful convergence is obtained. False otherwise.

    """
    enumerating = solve_data.active_strategy == 'enumerate'
    num_discrete_solns = None
    if enumerating:
        num_discrete_solns = len(
            solve_data.linear_GDP.GDPopt_utils.discrete_realizations)

    while solve_data.master_iteration < config.iterlim:
        # Set iteration counters for new master iteration.
        solve_data.master_iteration += 1
        solve_data.mip_iteration = 0
        solve_data.nlp_iteration = 0

        if enumerating:
            # Warn people if they're insane
            config.logger.info('---GDPopt Enumeration Iteration %s of %s---' %
                               (solve_data.master_iteration,
                                num_discrete_solns))

            # 'fix' a master solution so that master will be a feasibility
            # problem
            _fix_discrete_solution(solve_data, solve_data.master_iteration - 1)

        else:
            # print line for visual display
            config.logger.info(
                '---GDPopt Master Iteration %s---'
                % solve_data.master_iteration)

        # solve linear master problem
        with time_code(solve_data.timing, 'mip'):
            mip_result = solve_LOA_master(solve_data, config)

        # Check termination conditions
        if algorithm_should_terminate(solve_data, config):
            break

        # go to next discrete solution without solving the subproblem, if this
        # one wasn't feasible.
        if enumerating and not mip_result.feasible:
            if solve_data.master_iteration < num_discrete_solns:
                continue
            else: # we're done
                _close_enumeration_bounds(solve_data)
                break

        # Solve NLP subproblem
        if solve_data.active_strategy == 'LOA':
            with time_code(solve_data.timing, 'nlp'):
                nlp_result = solve_local_subproblem(mip_result, solve_data,
                                                    config)
            if nlp_result.feasible:
                add_outer_approximation_cuts(nlp_result, solve_data, config)
        elif solve_data.active_strategy == 'GLOA':
            with time_code(solve_data.timing, 'nlp'):
                nlp_result = solve_global_subproblem(mip_result, solve_data,
                                                     config)
            if nlp_result.feasible:
                add_affine_cuts(nlp_result, solve_data, config)
        elif solve_data.active_strategy == 'RIC':
            with time_code(solve_data.timing, 'nlp'):
                nlp_result = solve_local_subproblem(mip_result, solve_data,
                                                    config)
        elif enumerating:
            with time_code(solve_data.timing, 'nlp'):
                nlp_result = solve_global_subproblem(mip_result, solve_data,
                                                     config)
        else:
            raise ValueError('Unrecognized strategy: ' +
                             solve_data.active_strategy)

        # Add integer cut
        if not enumerating:
            add_integer_cut(mip_result.var_values, solve_data.linear_GDP,
                            solve_data, config, feasible=nlp_result.feasible)

        # Check termination conditions
        if algorithm_should_terminate(solve_data, config, num_discrete_solns):
            break

def _close_enumeration_bounds(solve_data):
    if solve_data.best_solution_found is None:
        # the problem is infeasible: We never solved a subproblem
        solve_data.results.solver.termination_condition = tc.infeasible
        objective = next(
            solve_data.working_model.component_data_objects(Objective,
                                                            active=True))
        if objective.sense == minimize:
            solve_data.LB = float('inf')
        else:
            solve_data.UB = -float('inf')
    else:
        # we know the optimal solution: set the bounds accordingly
        obj_val = value(next(
            solve_data.best_solution_found.component_data_objects(
                Objective, active=True)))
        solve_data.LB = obj_val
        solve_data.UB = obj_val
        solve_data.results.solver.termination_condition = tc.optimal

def algorithm_should_terminate(solve_data, config, num_enumeration_iters=None):
    """Check if the algorithm should terminate.

    Termination conditions based on solver options and progress.

    """
    # Check bound convergence
    if solve_data.LB + config.bound_tolerance >= solve_data.UB:
        config.logger.info(
            'GDPopt exiting on bound convergence. '
            'LB: {:.10g} + (tol {:.10g}) >= UB: {:.10g}'.format(
                solve_data.LB, config.bound_tolerance, solve_data.UB))
        if solve_data.LB == float('inf') and solve_data.UB == float('inf'):
            solve_data.results.solver.termination_condition = tc.infeasible
        elif solve_data.LB == float('-inf') and solve_data.UB == float('-inf'):
            solve_data.results.solver.termination_condition = tc.infeasible
        else:
            solve_data.results.solver.termination_condition = tc.optimal
        return True

    # Check if we're done enumerating
    if num_enumeration_iters is not None and \
       solve_data.master_iteration >= num_enumeration_iters:
        _close_enumeration_bounds(solve_data)
        return True

    # Check iteration limit
    if solve_data.master_iteration >= config.iterlim:
        config.logger.info(
            'GDPopt unable to converge bounds '
            'after %s master iterations.'
            % (solve_data.master_iteration,))
        config.logger.info(
            'Final bound values: LB: {:.10g}  UB: {:.10g}'.format(
                solve_data.LB, solve_data.UB))
        solve_data.results.solver.termination_condition = tc.maxIterations

        return True

    # Check time limit
    elapsed = get_main_elapsed_time(solve_data.timing)
    if elapsed >= config.time_limit:
        config.logger.info(
            'GDPopt unable to converge bounds '
            'before time limit of {} seconds. '
            'Elapsed: {} seconds'
            .format(config.time_limit, elapsed))
        config.logger.info(
            'Final bound values: LB: {}  UB: {}'.
            format(solve_data.LB, solve_data.UB))
        solve_data.results.solver.termination_condition = tc.maxTimeLimit
        return True

    if not algorithm_is_making_progress(solve_data, config):
        config.logger.debug(
            'Algorithm is not making enough progress. '
            'Exiting iteration loop.')
        solve_data.results.solver.termination_condition = tc.locallyOptimal
        return True
    return False


def algorithm_is_making_progress(solve_data, config):
    """Make sure that the algorithm is making sufficient progress
    at each iteration to continue."""

    # TODO if backtracking is turned on, and algorithm visits the same point
    # twice without improvement in objective value, turn off backtracking.

    # TODO stop iterations if feasible solutions not progressing for a number
    # of iterations.

    # If the hybrid algorithm is not making progress, switch to OA.
    # required_feas_prog = 1E-6
    # if solve_data.working_model.GDPopt_utils.objective.sense == minimize:
    #     sign_adjust = 1
    # else:
    #     sign_adjust = -1

    # Maximum number of iterations in which feasible bound does not
    # improve before terminating algorithm
    # if (len(feas_prog_log) > config.algorithm_stall_after and
    #     (sign_adjust * (feas_prog_log[-1] + required_feas_prog)
    #      >= sign_adjust *
    #      feas_prog_log[-1 - config.algorithm_stall_after])):
    #     config.logger.info(
    #         'Feasible solutions not making enough progress '
    #         'for %s iterations. Algorithm stalled. Exiting.\n'
    #         'To continue, increase value of parameter '
    #         'algorithm_stall_after.'
    #         % (config.algorithm_stall_after,))
    #     return False

    return True
