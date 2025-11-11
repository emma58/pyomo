#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.solver.solvers.gurobi.gurobi_persistent import GurobiPersistent
from pyomo.contrib.solver.solvers.gurobi.gurobi_direct import GurobiDirect
from pyomo.contrib.solver.solvers.gurobi.gurobi_direct_minlp import GurobiDirectMINLP
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import TerminationCondition
from pyomo.core.expr.taylor_series import taylor_series_expansion
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    ConstraintList,
    Integers,
    Binary,
    maximize,
)

parameterized, param_available = attempt_import('parameterized')
parameterized = parameterized.parameterized

gurobipy, gurobipy_available = attempt_import('gurobipy')


if not param_available:
    raise unittest.SkipTest('Parameterized is not available.')


gurobi_minlp = SolverFactory('gurobi_direct_minlp')
gurobi_direct = SolverFactory('gurobi_direct')
gurobi_persistent = SolverFactory('gurobi_persistent')

all_gurobis = [
    #('gurobi_minlp', gurobi_minlp),
    #('gurobi_direct', gurobi_direct),
    ('gurobi_persistent', gurobi_persistent),
]


def _load_tests(solver_list):
    res = list()
    for solver_name, solver in solver_list:
        res.append((f"{solver_name}", solver))
    return res


@unittest.skipUnless(gurobipy_available, 'gurobipy is not available')
class TestCallbacks(unittest.TestCase):
    @parameterized.expand(input=_load_tests(all_gurobis))
    def test_lazy_cut_callback(self, name, opt):
        if not opt.available():
            raise unittest.SkipTest(f'Solver {opt.name} not available')

        m = ConcreteModel()
        m.x = Var(bounds=(0, 4))
        m.y = Var(within=Integers, bounds=(0, None))
        m.obj = Objective(expr=2 * m.x + m.y)
        m.cons = ConstraintList()

        def _add_cut(xval):
            m.x.value = xval
            return m.cons.add(m.y >= taylor_series_expansion((m.x - 2) ** 2))

        _add_cut(0)
        _add_cut(4)

        if name == 'gurobi_persistent':
            opt.set_instance(m)

        opt.config.solver_options['PreCrush'] = 1
        opt.config.solver_options['LazyConstraints'] = 1
        # opt.set_gurobi_param('PreCrush', 1)
        # opt.set_gurobi_param('LazyConstraints', 1)

        def _my_callback(cb_m, cb_opt, cb_where):
            if cb_where == gurobipy.GRB.Callback.MIPSOL:
                cb_opt.cbGetSolution(variables=[m.x, m.y])
                if m.y.value < (m.x.value - 2) ** 2 - 1e-6:
                    cb_opt.cbLazy(_add_cut(m.x.value))

        opt.set_callback(_my_callback)
        opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)

    @parameterized.expand(input=_load_tests(all_gurobis))
    def test_cut_off_at_root_callback(self, name, opt):
        m = ConcreteModel()
        m.x = Var([0, 1], domain=Binary)
        m.cons = Constraint(expr=m.x[0] + m.x[1] == 1)
        # @m.Constraint([0, 1])
        # def cons(m, i):
        #     return m.x[i] <= 0.5

        m.obj = Objective(expr= m.x[0] + m.x[1], sense=maximize)

        def _terminate_after_root(cb_model, cb_opt, cb_where):
            if cb_where == gurobipy.GRB.Callback.MIPNODE:
                status = cb_opt.cbGet(gurobipy.GRB.Callback.MIPNODE_STATUS)
                if status == gurobipy.GRB.OPTIMAL:
                    # load LP relaxation solution into the pyomo model
                    cb_opt.cbGetNodeRel(cb_model.all_var_list)
                    node_count = cb_opt.cbGet(gurobipy.GRB.Callback.MIPNODE_NODCNT)
                    assert node_count == 0.0 # Why is this a float, Gurobi?
                    cb_opt._solver_model.terminate()

        opt.set_callback(_terminate_after_root)
        opt.config.solver_options['Presolve'] = 0
        opt.config.solver_options['Heuristics'] = 0
        opt.config.solver_options['Cuts'] = 0
        opt.config.solver_options['CutPasses'] = 0
        opt.config.solver_options['PreCrush'] = 1
        opt.config.solver_options['Threads'] = 1
        opt.config.solver_options['PreDual'] = 0
        opt.config.solver_options['OBBT'] = 0

        results = opt.solve(m, tee=True)
        self.assertEqual(results.termination_condition,
                         TerminationCondition.interrupted)
        self.assertEqual(value(m.x[0], 0.5))
        self.assertEqual(value(m.x[1], 0.5))
