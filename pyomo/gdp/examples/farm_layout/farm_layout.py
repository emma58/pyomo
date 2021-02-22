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

# Farm Layout problem from TODO
# nonlinear convex GDP

def instantiate_farm_layout(farm_data):
    m = ConcreteModel()

    ## Sets
    m.Rectangles = Set(initialize=farm_data['rectangles'])
    m.RectanglePairs = Set(initialize=m.Rectangles*m.Rectangles, 
                           filter=lambda _, r1, r2: r1 < r2)

    ## Params
    m.Area = Param(m.Rectangles, initialize=farm_data['area'])

    m.LengthLB = Param(m.Rectangles, initialize=farm_data['min_length'])
    def length_ub(m, r):
        return m.Area[r]/m.LengthLB[r]
    m.LengthUB = Param(m.Rectangles, initialize=length_ub)
    m.WidthLB = Param(m.Rectangles, initialize=farm_data['min_width'])
    def width_ub(m, r):
        return m.Area[r]/m.WidthLB[r]
    m.WidthUB = Param(m.Rectangles, initialize=width_ub)
    m.TotalLengthUB = Param(initialize=farm_data['total_length_ub'])
    m.TotalWidthUB = Param(initialize=farm_data['total_width_ub'])

    ## Vars
    def length_bounds(m, r):
        return (m.LengthLB[r], m.LengthUB[r])
    m.length = Var(m.Rectangles, bounds=length_bounds)

    def width_bounds(m, r):
        return (m.WidthLB[r], m.WidthUB[r])
    m.width = Var(m.Rectangles, bounds=width_bounds)

    m.total_length = Var(bounds=(0, m.TotalLengthUB))
    m.total_width = Var(bounds=(0, m.TotalWidthUB))

    def x_coord_bounds(m, r):
        return (0, m.TotalLengthUB - m.LengthLB[r])
    m.x_coord = Var(m.Rectangles, bounds=x_coord_bounds)

    def y_coord_bounds(m, r):
        return (0, m.TotalWidthUB - m.WidthLB[r])
    m.y_coord = Var(m.Rectangles, bounds=y_coord_bounds)

    ## Objective : minimize perimeter
    m.obj = Objective(expr=2*(m.total_length + m.total_width))

    ## Constraints
    @m.Constraint(m.Rectangles)
    def total_length_defn(m, r):
        return m.total_length >= m.x_coord[r] + m.length[r]

    @m.Constraint(m.Rectangles)
    def total_width_defn(m, r):
        return m.total_width >= m.y_coord[r] + m.width[r]

    @m.Constraint(m.Rectangles)
    def enforce_rectangle_area(m, r):
        return m.Area[r]/m.width[r] - m.length[r] <= 0

    @m.Disjunct(m.RectanglePairs)
    def LeftOf(d, i, j):
        m = d.model()
        d.no_overlap = Constraint(expr=m.x_coord[i] + m.length[i] <= \
                                  m.x_coord[j])

    @m.Disjunct(m.RectanglePairs)
    def RightOf(d, i, j):
        m = d.model()
        d.no_overlap = Constraint(expr=m.x_coord[j] + m.length[j] <= \
                                  m.x_coord[i])

    @m.Disjunct(m.RectanglePairs)
    def Above(d, i, j):
        m = d.model()
        d.no_overlap = Constraint(expr=m.y_coord[j] + m.width[j] <= m.y_coord[i])

    @m.Disjunct(m.RectanglePairs)
    def Below(d, i, j):
        m = d.model()
        d.no_overlap = Constraint(expr=m.y_coord[i] + m.width[i] <= m.y_coord[j])

    @m.Disjunction(m.RectanglePairs)
    def no_overlap(m, i, j):
        return [m.LeftOf[i,j], m.RightOf[i,j], m.Above[i,j], m.Below[i,j]]

    return m

def read_json(filename):
    with open(filename) as f:
        farm_dict = json.load(f)

    data = {}
    rectangles = data['rectangles'] = range(1, farm_dict['num_rectangles'] + 1)
    data['total_width_ub'] = farm_dict['total_width_ub']
    data['total_length_ub'] = farm_dict['total_length_ub']
    data['area'] = {}
    data['min_length'] = {}
    data['min_width'] = {}
    for rec in rectangles:
        data['area'][rec] = farm_dict['area'][str(rec)]
        data['min_length'][rec] = farm_dict['min_length'][str(rec)]
        data['min_width'][rec] = farm_dict['min_width'][str(rec)]
    
    return data

def instantiate_model(filename):
    data = read_json(filename)
    m = instantiate_farm_layout(data)

    return m

if __name__ == "__main__":
    m = instantiate_model('farm_layout_logcabin9.json')

    bigm = TransformationFactory('gdp.bigm').create_using(m)
    rbigm = TransformationFactory('core.relax_integer_vars').create_using(bigm)
    results = SolverFactory('ipopt').solve(rbigm, tee=True)

    # Reference(bigm.Above[...].indicator_var).pprint()
    # Reference(bigm.Below[...].indicator_var).pprint()
    # Reference(bigm.RightOf[...].indicator_var).pprint()
    # Reference(bigm.LeftOf[...].indicator_var).pprint()

    hull = TransformationFactory('gdp.hull').create_using(m)
    rhull = TransformationFactory('core.relax_integer_vars').create_using(hull)

    results = SolverFactory('ipopt').solve(rhull, tee=True)

    # Reference(hull.Above[...].indicator_var).pprint()
    # Reference(hull.Below[...].indicator_var).pprint()
    # Reference(hull.RightOf[...].indicator_var).pprint()
    # Reference(hull.LeftOf[...].indicator_var).pprint()

    set_trace()
