from pyomo.environ import *
from pyomo.gdp import *

import json

## DEBUG
from pdb import set_trace

def instantiate_retrofit_synthesis(rsyn_data):
    model = ConcreteModel()
    
    ## Sets
    model.S = Set(initialize=rsyn_data['streams'])
    model.S_PROD = Set(initialize=rsyn_data['product_streams_retrofit'],
                       within=model.S)
    model.S_RAW = Set(initialize=rsyn_data['raw_feed_streams_retrofit'],
                       within=model.S)
    model.T = Set(initialize=rsyn_data['time_periods'])
    model.P = Set(initialize=rsyn_data['processes'])
    model.M = Set(initialize=rsyn_data['scenarios'])
    model.I = Set(initialize=rsyn_data['process_units'])
    model.K = Set(initialize=rsyn_data['number_of_streams'])

    # ESJ: At the risk of getting too clever too soon:
    model.Nodes = Set(initialize=rsyn_data['nodes'])
    model.StreamsIntoProcess = Set(model.P,
                                   initialize=rsyn_data['streams_into_process'],
                                   within=model.S)
    model.StreamsOutOfProcess = Set(model.P,
                                    initialize=rsyn_data[
                                        'streams_out_of_process'],
                                    within=model.S)
    model.StreamsIntoNode = Set(model.Nodes,
                                initialize=rsyn_data['streams_into_node'],
                                within=model.S)
    model.StreamsOutOfNode = Set(model.Nodes,
                                 initialize=rsyn_data['streams_out_of_node'],
                                 within=model.S)

    ## Params
    model.Price_Prod = Param(model.S_PROD, model.T,
                         initialize=rsyn_data['prod_stream_prices'])
    model.Price_Raw = Param(model.S_RAW, model.T,
                            initialize=rsyn_data['raw_stream_prices'])
    model.FixedCost_ConvCap = Param(model.P, model.M, model.T,
                                    initialize=rsyn_data[
                                        'retrofit_conversion_fixed_costs'])
    model.FC = Param(model.I, model.T, initialize=rsyn_data['fixed_costs'])
    model.PC = Param(model.K, model.T, initialize=rsyn_data['cost_coefs'])
    model.MolWeight = Param(model.S, initialize=rsyn_data['molecular_weight'])
    model.Dem = Param(model.S_PROD, model.T, initialize=rsyn_data['demand'])
    model.Sup = Param(model.S_RAW, model.T, initialize=rsyn_data['supply'])
    model.Gamma = Param(model.S, initialize=rsyn_data[
        'stoichiometric_coefficients'])

    ## Vars
    model.mf = Var(model.S, model.T, within=NonNegativeReals)
    model.X = Var(model.K, model.T, within=NonNegativeReals)
    model.f = Var(model.S, model.T, within=NonNegativeReals)
    # TODO: whoops, this is going to be an indicator variable
    model.w = Var(model.P, model.M, model.T, within=Binary)
    model.R = Var(model.I, model.T, within=Binary)

    ## Objective
    model.obj = Objective(expr=sum(model.Price_Prod[s,t]*model.mf[s,t] for s in
                                   model.S_PROD for t in model.T) -
                          sum(model.Price_Raw[s,t]*model.mf[s,t] for s in
                              model.S_RAW for t in model.T) -
                          sum(model.FixedCost_ConvCap[p,m,t]*model.w[p,m,t]
                              for p in model.P for m in model.M for t in
                              model.T) + sum(model.FC[i,t]*model.R[i,t] for i
                                             in model.I for t in model.T) +
                          sum(model.PC[k,t]*model.X[k,t] for k in model.K
                              for t in model.T))

    ## Global Constraints
    @model.Constraint(model.S, model.T)
    def Equivalence_Mass(m, s, t):
        return m.mf[s,t] == m.f[s,t]*m.MolWeight[s]

    @model.Constraint(model.S_PROD, model.T)
    def Demand(m, s, t):
        return m.mf[s,t] >= m.Dem[s,t]

    @model.Constraint(model.S_RAW, model.T)
    def Supply(m, s, t):
        return m.mf[s,t] <= m.Sup[s,t]

    @model.Constraint(model.Nodes, model.T)
    def MassBalance_Node(m, n, t):
        return sum(m.mf[s,t] for s in m.StreamsIntoNode[n]) == \
            sum(m.mf[s,t] for s in m.StreamsOutOfNode[n])

    @model.Constraint(model.P, model.T)
    def MassBalance_Process(m, p, t):
        return sum(m.mf[s,t] for s in m.StreamsIntoProcess[p]) == \
            sum(m.mf[s,t] for s in m.StreamsOutOfProcess[p])

    # These are some hard-coded constraints "to ensure that conversion in every
    # process is being done relative to limiting reactant (We assume limiting
    # reactant is known apriori for every process)". It looks like we only need
    # them for the processes with multiple input streams.
    @model.Constraint(model.T)
    def Limiting_Process2(m, t):
        return m.f[3,t]*m.Gamma[3] <= m.f[7,t]*m.Gamma[7]

    @model.Constraint(model.T)
    def Limiting_Process6(m, t):
        return m.f[16,t]*m.Gamma[16] <= m.f[23,t]*m.Gamma[23]

    @model.Constraint(model.T)
    def Limiting_Process7(m, t):
        return m.f[24,t]*m.Gamma[24] <= m.f[31,t]*m.Gamma[31]

    ## Disjunctions

    return model

def read_json(filename):
    with open(filename) as f:
        syndict = json.load(f)

    data = {}
    data['product_streams_retrofit'] = syndict['product_streams_retrofit']
    data['raw_feed_streams_retrofit'] = syndict['raw_feed_streams_retrofit']
    data['time_periods'] = range(1, syndict['num_time_periods'] + 1)
    data['processes'] = range(1, syndict['num_processes'] + 1)
    data['scenarios'] = range(1, syndict['num_scenarios'] + 1)
    data['process_units'] = range(1, syndict['num_process_units'] + 1)
    data['number_of_streams'] = range(1, syndict['num_num_streams'] + 1)
    data['streams'] = range(1, syndict['num_streams'] + 1)
    data['nodes'] = range(1, syndict['num_nodes'] + 1)
    data['streams_into_process'] = {}
    data['streams_out_of_process'] = {}
    data['streams_into_node'] = {}
    data['streams_out_of_node'] = {}
    data['raw_stream_prices'] = {}
    data['prod_stream_prices'] = {}
    data['retrofit_conversion_fixed_costs'] = {}
    data['fixed_costs'] = {}
    data['cost_coefs'] = {}
    data['molecular_weight'] = {}
    data['demand'] = {}
    data['supply'] = {}
    data['stoichiometric_coefficients'] = {}
    for p in data['processes']:
        data['streams_into_process'][p] = syndict['streams_into_process'][str(p)]
        data['streams_out_of_process'][p] = syndict['streams_out_of_process'][
            str(p)]
    for n in data['nodes']:
        data['streams_into_node'][n] = syndict['streams_into_node'][str(n)]
        data['streams_out_of_node'][n] = syndict['streams_out_of_node'][str(n)]
    for t in data['time_periods']:
        for s in data['raw_feed_streams_retrofit']:
            data['raw_stream_prices'][(s,t)] = syndict['raw_stream_prices'][
                str((s,t))]
            data['supply'][(s,t)] = syndict['supply'][str((s,t))]
        for s in data['product_streams_retrofit']:
            data['prod_stream_prices'][(s,t)] = syndict['prod_stream_prices'][
                str((s,t))]
            data['demand'][(s,t)] = syndict['demand'][str((s,t))]
        for p in data['processes']:
            for m in data['scenarios']:
                data['retrofit_conversion_fixed_costs'][(p,m,t)] = syndict[
                    'retrofit_conversion_fixed_costs'][str((p,m,t))]
        for i in data['process_units']:
            data['fixed_costs'][(i,t)] = syndict['fixed_costs'][str((i,t))]
        for k in data['number_of_streams']:
            data['cost_coefs'][(k,t)] = syndict['cost_coefs'][str((k,t))]
        first_time = False
    for s in data['streams']:
        data['molecular_weight'][s] = syndict['molecular_weight'][str(s)]
        data['stoichiometric_coefficients'][s] = syndict[
            'stoichiometric_coefficients'][str(s)]

    return data

def instantiate_model(filename):
    data = read_json(filename)
    m = instantiate_retrofit_synthesis(data)

    return m

if __name__ == "__main__":
    m = instantiate_model('rsyn084004.json')

    TransformationFactory('gdp.bigm').apply_to(m)
    results = SolverFactory('gurobi').solve(m, tee=True)
    set_trace()
