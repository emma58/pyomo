#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from __future__ import division

from pyomo.environ import *
from pyomo.gdp import *

import json

## DEBUG
from pdb import set_trace

# Safety layout problem (I think) from Nick Sawaya's thesis
# nonlinear convex GDP

def instantiate_safety_layout(safety_data):
    m = ConcreteModel()

    ## Sets
    m.Rectangles = Set(initialize=safety_data['rectangles'])
    m.RectanglePairs = Set(initialize=m.Rectangles*m.Rectangles, 
                           filter=lambda _, r1, r2: r1 < r2)

    ## Parameters
    m.Length = Param(m.Rectangles, initialize=safety_data['length'])    
    m.Height = Param(m.Rectangles, initialize=safety_data['height'])
    m.Cost = Param(m.Rectangles, initialize=safety_data['cost'])

    m.SafeX = Param(m.Rectangles, initialize=safety_data['x'])
    m.SafeY = Param(m.Rectangles, initialize=safety_data['y'])

    m.PWCost = Param(m.RectanglePairs, initialize=safety_data['pairwise_costs'])

    # scalars
    m.TotalLength = Param(initialize=safety_data['total_length'])    
    m.TotalWidth = Param(initialize=safety_data['total_width'])

    ## Variables
    def x_bounds_rule(m, i):
        return (m.Length[i]/2, m.TotalLength - m.Length[i]/2)
    m.x = Var(m.Rectangles, bounds=x_bounds_rule)

    def y_bounds_rule(m, i):
        return (m.Height[i]/2, m.TotalWidth - m.Height[i]/2)
    m.y = Var(m.Rectangles, bounds=y_bounds_rule)

    m.x_dist = Var(m.RectanglePairs, within=NonNegativeReals)
    m.y_dist = Var(m.RectanglePairs, within=NonNegativeReals)

    ## Objective
    m.obj = Objective(expr=sum(m.PWCost[i,j]*(m.x_dist[i,j] + m.y_dist[i,j]) for
                               i,j in m.RectanglePairs) + \
                      sum(m.Cost[i]*((m.x[i] - m.SafeX[i])**2 + \
                          (m.y[i] - m.SafeY[i])**2) for i in m.Rectangles))

    ## Constraints
    @m.Constraint(m.RectanglePairs)
    def x_distance1(m, i, j):
        return m.x_dist[i,j] >= m.x[i] - m.x[j]

    @m.Constraint(m.RectanglePairs)
    def x_distance2(m, i, j):
        return m.x_dist[i,j] >= m.x[j] - m.x[i]

    @m.Constraint(m.RectanglePairs)
    def y_distance1(m, i, j):
        return m.y_dist[i,j] >= m.y[i] - m.y[j]

    @m.Constraint(m.RectanglePairs)
    def y_distance2(m, i, j):
        return m.y_dist[i,j] >= m.y[j] - m.y[i]

    ## Disjunctions
    @m.Disjunct(m.RectanglePairs)
    def LeftOf(d, i, j):
        m = d.model()
        d.no_overlap = Constraint(expr=m.x[i] + m.Length[i]/2 <= \
                                  m.x[j] - m.Length[j]/2)

    @m.Disjunct(m.RectanglePairs)
    def RightOf(d, i, j):
        m = d.model()
        d.no_overlap = Constraint(expr=m.x[j] + m.Length[j]/2 <= \
                                  m.x[i] - m.Length[i]/2)

    @m.Disjunct(m.RectanglePairs)
    def Above(d, i, j):
        m = d.model()
        d.no_overlap = Constraint(expr=m.y[j] + m.Height[j]/2 <= \
                                  m.y[i] - m.Height[i]/2)

    @m.Disjunct(m.RectanglePairs)
    def Below(d, i, j):
        m = d.model()
        d.no_overlap = Constraint(expr=m.y[i] + m.Height[i]/2 <= \
                                  m.y[j] - m.Height[j]/2)

    @m.Disjunction(m.RectanglePairs)
    def no_overlap(m, i, j):
        return [m.LeftOf[i,j], m.RightOf[i,j], m.Above[i,j], m.Below[i,j]]

    return m

def read_json(filename):
    with open(filename) as f:
        safety_dict = json.load(f)

    data = {}
    rectangles = data['rectangles'] = range(1, safety_dict['num_rectangles'] + 1)
    data['total_width'] = safety_dict['total_width']
    data['total_length'] = safety_dict['total_length']
    data['cost'] = {}
    data['length'] = {}
    data['height'] = {}
    data['x'] = {}
    data['y'] = {}
    data['pairwise_costs'] = {}
    for rec in rectangles:
        data['cost'][rec] = safety_dict['cost'][str(rec)]
        data['length'][rec] = safety_dict['length'][str(rec)]
        data['height'][rec] = safety_dict['height'][str(rec)]
        data['x'][rec] = safety_dict['x'][str(rec)]
        data['y'][rec] = safety_dict['y'][str(rec)]
        for j in rectangles:
            if j > rec:
                data['pairwise_costs'][(rec,j)] = safety_dict[
                    'pairwise_costs']['(%s,%s)' % (rec, j)]
    return data

def instantiate_model(filename):
    data = read_json(filename)
    m = instantiate_safety_layout(data)

    return m

if __name__ == "__main__":
    m = instantiate_model('safety_layout4.json')

    TransformationFactory('gdp.bigm').apply_to(m)
    results = SolverFactory('gurobi').solve(m, tee=True)
    set_trace()
