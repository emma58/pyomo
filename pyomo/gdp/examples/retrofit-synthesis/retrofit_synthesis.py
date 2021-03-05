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
    model.Eta = Param(model.P, model.M, model.T,
                      initialize=rsyn_data['conversion_rate'])
    model.Cap = Param(model.P, model.M, model.T,
                      initialize=rsyn_data['capacity_limit'])
    model.InvestmentLimit = Param(model.T,
                                  initialize=rsyn_data['investment_limit'])

    ## Vars
    # mass flowrate
    model.mf = Var(model.S, model.T, within=NonNegativeReals)
    model.X = Var(model.K, model.T, within=NonNegativeReals)
    # these have some random hard-coded bounds:
    for t in model.T:
        model.X[1,t].setub(10)
        model.X[12,t].setub(7)
        model.X[29,t].setub(7)
        model.X[30,t].setub(5)
        model.X[57,t].setub(7)
        model.X[74,t].setub(7)
        model.X[75,t].setub(5)

    # molar flowrate
    model.f = Var(model.S, model.T, within=NonNegativeReals)
    # mass flowrate of unreacted streams for each processs
    model.mf_unreacted = Var(model.P, model.T, within=NonNegativeReals)
    model.convcapcost = Var(model.P, model.T, within=NonNegativeReals)
    model.COST = Var(model.I, model.T, within=Reals)

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
            sum(m.mf[s,t] for s in m.StreamsOutOfProcess[p]) + \
            m.mf_unreacted[p,t]

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
    @model.Disjunct(model.P, model.M, model.T)
    def Disj1(d, p, m, t):
        # indicator var y from gams
        model = d.model()
        i = model.StreamsIntoProcess[p].first()
        @d.Constraint(model.StreamsOutOfProcess[p])
        def conv(d, s):
            return model.f[s,t] <= model.f[i,t]*(model.Gamma[s]/model.Gamma[i])*\
                model.Eta[p,m,t]

        d.cap = Constraint(expr=sum(model.mf[s,t] for s in
                                    model.StreamsIntoProcess[p]) <= \
                           model.Cap[p,m,t])

    @model.Disjunction(model.P, model.T)
    def Disjunction1(model, p, t):
        return [model.Disj1[p,m,t] for m in model.M]

    @model.Disjunct(model.P, model.M, model.T)
    def Disj2(d, p, m, t):
        # indicator var w from gams
        model = d.model()
        d.convcapcost = Constraint(expr=model.convcapcost[p,t] == \
                                   model.FixedCost_ConvCap[p,m,t])

    @model.Disjunction(model.P, model.T)
    def Disjunction2(model, p, t):
        return [model.Disj2[p,m,t] for m in model.M]

    @model.Constraint(model.T)
    def LimitingCost(m, t):
        return 1e3*sum(m.convcapcost[p,t] for p in m.P) + \
            sum(m.Price_Raw[s,t]*m.mf[s,t] for s in m.S_RAW) <= \
            1e3*m.InvestmentLimit[t]

    ## Logical Constraints
    @model.Constraint(model.P, model.M, model.T, model.T)
    def Logic_y(model, p, m, t, tau):
        if m == 1 or t >= tau:
            return Constraint.Skip
        return model.Disj1[p,m,t].indicator_var <= \
            model.Disj1[p,m,tau].indicator_var

    @model.Constraint(model.P, model.M, model.T, model.T)
    def Logic_w(model, p, m, t, tau):
        if m == 1 or t == tau:
            return Constraint.Skip
        return model.Disj2[p,m,t].indicator_var <= \
            model.Disj2[p,1,tau].indicator_var

    @model.Constraint(model.P, model.T)
    def Logic_yw_M1(model, p, t):
        return model.Disj1[p, 1, t].indicator_var <= \
            model.Disj1[p, 1, t].indicator_var

    # ESJ TODO: I'm baffled... This should only work for t = 4??
    # @model.Constraint(model.P, model.M, model.T)
    # def Logic_yw_MLessM1(model, p, m, t):
    #     if m == 1:
    #         return Constraint.Skip
    #     return model.Disj1[p,m,t].indicator_var <= \
    #         model.Disj1[p,m,t].indicator_var + \
    #         model.Disj1[p,m,t-1].indicator_var + \
    #         model.Disj1[p,m,t-2].indicator_var + \
    #         model.Disj1[p,m,t-3].indicator_var

    ## Interconnecting Streams
    @model.Constraint(model.T)
    def Inter1(model, t):
        return model.mf[5,t] == model.mf[32,t] + model.X[1,t]
    @model.Constraint(model.T)
    def Inter2(model, t):
        return model.mf[9,t] == model.mf[33,t] + model.X[12,t]
    @model.Constraint(model.T)
    def Inter3(model, t):
        return model.mf[20,t] == model.mf[34,t] + model.X[29,t]
    @model.Constraint(model.T)
    def Inter4(model, t):
        return model.mf[21,t] == model.mf[35,t] + model.X[30,t]

    ## Mass Balances
    @model.Constraint(model.T)
    def MB1(model, t):
        return model.X[1,t] == model.X[2,t] + model.X[3,t]
    @model.Constraint(model.T)
    def MB2(model, t):
        return model.X[6,t] == model.X[4,t] + model.X[5,t]
    @model.Constraint(model.T)
    def MB3(model, t):
        return model.X[6,t] == model.X[7,t] + model.X[8,t]
    @model.Constraint(model.T)
    def MB4(model, t):
        return model.X[8,t] == model.X[9,t] + model.X[10,t] + model.X[11,t]
    @model.Constraint(model.T)
    def MB5(model, t):
        return model.X[13,t] == model.X[16,t] + model.X[17,t]
    @model.Constraint(model.T)
    def MB6(model, t):
        return model.X[15,t] == model.X[18,t] + model.X[19,t] + model.X[20,t]
    @model.Constraint(model.T)
    def MB7(model, t):
        return model.X[23,t] == model.X[27,t] + model.X[28,t]
    @model.Constraint(model.T)
    def MB8(model, t):
        return model.X[31,t] == model.X[24,t] + model.X[30,t]
    @model.Constraint(model.T)
    def MB9(model, t):
        return model.X[25,t] == model.X[32,t] + model.X[33,t]
    @model.Constraint(model.T)
    def MB10(model, t):
        return model.X[26,t] == model.X[34,t] + model.X[35,t] + model.X[36,t]

    ## Connecting Stream between flowsheets
    @model.Constraint(model.T)
    def MB11(model, t):
        return model.X[45,t] == model.X[46,t]

    # ESJ: These are identical to MB1 through MB10, but offset by 45. Why?!
    @model.Constraint(model.T)
    def MB12(model, t):
        return model.X[46,t] == model.X[47,t] + model.X[48,t]
    @model.Constraint(model.T)
    def MB13(model, t):
        return model.X[51,t] == model.X[49,t] + model.X[50,t]
    @model.Constraint(model.T)
    def MB14(model, t):
        return model.X[51,t] == model.X[52,t] + model.X[53,t]
    @model.Constraint(model.T)
    def MB15(model, t):
        return model.X[53,t] == model.X[54,t] + model.X[55,t] + model.X[56,t]
    @model.Constraint(model.T)
    def MB16(model, t):
        return model.X[58,t] == model.X[61,t] + model.X[62,t]
    @model.Constraint(model.T)
    def MB17(model, t):
        return model.X[60,t] == model.X[63,t] + model.X[64,t] + model.X[65,t]
    @model.Constraint(model.T)
    def MB18(model, t):
        return model.X[68,t] == model.X[72,t] + model.X[73,t]
    @model.Constraint(model.T)
    def MB19(model, t):
        return model.X[76,t] == model.X[69,t] + model.X[75,t]
    @model.Constraint(model.T)
    def MB20(model, t):
        return model.X[70,t] == model.X[77,t] + model.X[78,t]
    @model.Constraint(model.T)
    def MB21(model, t):
        return model.X[71,t] == model.X[79,t] + model.X[80,t] + model.X[81,t]

    ## Disjunctions
    model.Z = Disjunction(model.I, model.T)
    for t in model.T:
        model.Z[1,t] = [[model.X[4,t] <= log(1 + model.X[2,t])], 
                        [model.X[2,t] <= 0, model.X[4,t] <= 0]]
        model.Z[2,t] = [[model.X[5,t] <= 1.2*log(1 + model.X[3,t])], 
                        [model.X[3,t] <= 0,model.X[5,t] <= 0] ]
        model.Z[3,t] = [[model.X[13,t] == 0.75*model.X[9,t]], 
                        [model.X[9,t] <= 0, model.X[13,t] <= 0]]
        model.Z[4,t] = [[model.X[14,t] <= 1.5*log(1 + model.X[10,t])], 
                        [model.X[10,t] <= 0, model.X[14,t] <= 0]]
        model.Z[5,t] = [[model.X[15,t] == model.X[11,t], 
                         model.X[15,t] == 0.5*model.X[12,t]], 
                        [model.X[11,t] <= 0, model.X[12,t] <= 0, 
                         model.X[15,t] <= 0]]
        model.Z[6,t] = [[model.X[21,t] <= 1.25*log(1 + model.X[16,t])], 
                        [model.X[16,t] <= 0, model.X[21,t] <= 0]]
        model.Z[7,t] = [[model.X[22,t] <= 0.9*log(1 + model.X[17,t])], 
                        [model.X[17,t] <= 0, model.X[22,t] <= 0]]
        model.Z[8,t] = [[model.X[23,t] <= log(1 + model.X[14,t])], 
                        [model.X[23,t] <= 0, model.X[14,t] <= 0]]
        model.Z[9,t] = [[model.X[24,t] == 0.9*model.X[18,t]], 
                        [model.X[18,t] <= 0, model.X[24,t] <= 0]]
        model.Z[10,t] = [[model.X[25,t] == 0.6*model.X[19,t]], 
                         [model.X[19,t] <= 0, model.X[25,t] <= 0]]
        model.Z[11,t] = [[model.X[26,t] <= 1.1*log(1 + model.X[20,t])], 
                         [model.X[20,t] <= 0, model.X[26,t] <= 0]]
        model.Z[12,t] = [[model.X[37,t] == 0.9*model.X[21,t], 
                          model.X[37,t] == model.X[39,t]], 
                         [model.X[21,t] <= 0, model.X[29,t] <= 0, 
                          model.X[37,t] <= 0]]
        model.Z[13,t] = [[model.X[38,t] <= log(1 + model.X[22,t])], 
                         [model.X[22,t] <= 0, model.X[38,t] <= 0]]
        model.Z[14,t] = [[model.X[39,t] <= 0.7*log(1 + model.X[27,t])], 
                         [model.X[27,t] <= 0, model.X[39,t] <= 0]]
        model.Z[15,t] = [[model.X[40,t] <= 0.65*log(1 + model.X[28,t]), 
                          model.X[40,t] <= 0.65*log(1 + model.X[31,t])], 
                         [model.X[28,t] <= 0, model.X[31,t] <= 0, 
                          model.X[40,t] <= 0]]
        model.Z[16,t] = [[model.X[41,t] == model.X[32,t]], 
                         [model.X[32,t] <= 0, model.X[41,t] <= 0]]
        model.Z[17,t] = [[model.X[42,t] == model.X[33,t]], 
                         [model.X[33,t] <= 0, model.X[42,t] <= 0]]
        model.Z[18,t] = [[model.X[43,t] <= 0.75*log(1 + model.X[34,t])], 
                         [model.X[34,t] <= 0, model.X[43,t] <= 0]]
        model.Z[19,t] = [[model.X[44,t] <= 0.8*log(1 + model.X[35,t])], 
                         [model.X[35,t] <= 0, model.X[44,t] <= 0]]
        model.Z[20,t] = [[model.X[45,t] <= 0.85*log(1 + model.X[36,t])], 
                         [model.X[36,t] <= 0, model.X[45,t] <= 0]]
        model.Z[21,t] = [[model.X[49,t] <= log(1 + model.X[47,t])], 
                         [model.X[47,t] <= 0, model.X[49,t] <= 0]]
        model.Z[22,t] = [[model.X[50,t] <= 1.2*log(1 + model.X[48,t])], 
                         [model.X[48,t] <= 0, model.X[50,t] <= 0]]
        model.Z[23,t] = [[model.X[58,t] == 0.75*model.X[54,t]], 
                         [model.X[58,t] <= 0, model.X[54,t] <= 0]]
        model.Z[24,t] = [[model.X[59,t] <= 1.5*log(1 + model.X[55,t])], 
                         [model.X[55,t] <= 0, model.X[59,t] <= 0]]
        model.Z[25,t] = [[model.X[60,t] == model.X[56,t], 
                          model.X[60,t] == 0.5*model.X[57,t]], 
                         [model.X[56,t] <= 0, model.X[57,t] <= 0, 
                          model.X[60,t] <= 0]]
        model.Z[26,t] = [[model.X[66,t] <= 1.25*log(1 + model.X[61,t])], 
                         [model.X[61,t] <= 0, model.X[66,t] <= 0]]
        model.Z[27,t] = [[model.X[67,t] <= 0.9*log(1 + model.X[62,t])], 
                         [model.X[62,t] <= 0, model.X[67,t] <= 0]]
        model.Z[28,t] = [[model.X[68,t] <= log(1 + model.X[59,t])], 
                         [model.X[59,t] <= 0, model.X[68,t] <= 0]]
        model.Z[29,t] = [[model.X[69,t] == 0.9*model.X[63,t]], 
                         [model.X[63,t] <= 0, model.X[69,t] <= 0]]
        model.Z[30,t] = [[model.X[70,t] == 0.6*model.X[64,t]], 
                         [model.X[64,t] <= 0, model.X[70,t] <= 0]]
        model.Z[31,t] = [[model.X[71,t] <= 1.1*log(1 + model.X[65,t])], 
                         [model.X[65,t] <= 0, model.X[71,t] <= 0]]
        model.Z[32,t] = [[model.X[82,t] == 0.9*model.X[66,t], 
                          model.X[82,t] == model.X[74,t]], 
                         [model.X[66,t] <= 0, model.X[74,t] <= 0, 
                          model.X[82,t] <= 0]]
        model.Z[33,t] = [[model.X[83,t] <= log(1 + model.X[67,t])], 
                         [model.X[67,t] <= 0, model.X[83,t] <= 0]]
        model.Z[34,t] = [[model.X[84,t] <= 0.7*log(1 + model.X[72,t])], 
                         [model.X[72,t] <= 0, model.X[84,t] <= 0]]
        model.Z[35,t] = [[model.X[85,t] <= 0.65*log(1 + model.X[73,t]), 
                          model.X[85,t] <= 0.65*log(1 + model.X[76,t])], 
                         [model.X[73,t] <= 0, model.X[76,t] <= 0, 
                          model.X[85,t] <= 0]]
        model.Z[36,t] = [[model.X[86,t] == model.X[77,t]], 
                         [model.X[77,t] <= 0, model.X[86,t] <= 0]]
        model.Z[37,t] = [[model.X[87,t] == model.X[78,t]], 
                         [model.X[78,t] <= 0, model.X[87,t] <= 0]]
        model.Z[38,t] = [[model.X[88,t] <= 0.75*log(1 + model.X[79,t])], 
                         [model.X[79,t] <= 0, model.X[88,t] <= 0]]
        model.Z[39,t] = [[model.X[89,t] <= 0.8*log(1 + model.X[80,t])], 
                         [model.X[80,t] <= 0, model.X[89,t] <= 0]]
        model.Z[40,t] = [[model.X[90,t] <= 0.85*log(1 + model.X[81,t])], 
                         [model.X[81,t] <= 0, model.X[90,t] <= 0]]

    @model.Disjunct(model.I, model.T)
    def RDisj(d, i, t):
        # indicator var R from gams
        m = d.model()
        d.cons = Constraint(expr=m.COST[i,t] == m.FC[i,t])
    @model.Disjunction(model.I)
    def Synthesis_Disj2(model, i):
        return [model.RDisj[i,t] for t in model.T]
    # ESJ: This is what's implemented in gams, but sure looks like an XOR over
    # time to me...
    # def logic_R_rule(m, i, t, tau):
    #     if t == tau:
    #         return Disjunction.Skip
    #     return [m.Synthesis_Disj2[i,t], m.Synthesis_Disj2[i,tau]]
    # model.Synthesis_Disjunction2 = Disjunction(model.I, model.T, model.T,
    #                                            rule=logic_R_rule, xor=False)

    @model.Constraint(model.I, model.T, model.T)
    def Logic_Z(model, i, t, tau):
        if t >= tau:
            return Constraint.Skip
        return model.Z[i,t].disjuncts[0].indicator_var <= \
            model.Z[i,tau].disjuncts[0].indicator_var

    # ESJ: This is the same question as above. I either don't understand gams or
    # it does something really creepy when the index is out of range...
    # @model.Constraint(model.I, model.T)
    # def Logic_ZR(model, i, t):
    #     return model.Z[i,t].disjuncts[0].indicator_var <= \
    #         model.RDisj[i,t].indicator_var + \
    #         model.Z[i,t-1].disjuncts[0].indicator_var + \
    #         model.Z[i,t-2].disjuncts[0].indicator_var + \
    #         model.Z[i,t-3].disjuncts[0].indicator_var

    ## design specifications
    @model.Constraint(model.T)
    def D1(model, t):
        return model.Z[1,t].disjuncts[0].indicator_var + \
            model.Z[2,t].disjuncts[0].indicator_var == 1

    ## TODO: There are 15 and a half million logic cuts now :,( Do we want them?

    ## Objective
    model.obj = Objective(expr=sum(model.Price_Prod[s,t]*model.mf[s,t] for s in
                                   model.S_PROD for t in model.T) -
                          sum(model.Price_Raw[s,t]*model.mf[s,t] for s
                              in model.S_RAW for t in model.T) -
                          sum(model.FixedCost_ConvCap[p,m,t]*\
                              model.Disj2[p,m,t].indicator_var for p in
                              model.P for m in model.M for t in model.T) +
                          sum(model.FC[i,t]*model.RDisj[i,t].indicator_var
                              for i in model.I for t in model.T) +
                          sum(model.PC[k,t]*model.X[k,t] for k in
                              model.K for t in model.T))

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
    data['conversion_rate'] = {}
    data['capacity_limit'] = {}
    data['investment_limit'] = {}
    for p in data['processes']:
        data['streams_into_process'][p] = syndict['streams_into_process'][str(p)]
        data['streams_out_of_process'][p] = syndict['streams_out_of_process'][
            str(p)]
    for n in data['nodes']:
        data['streams_into_node'][n] = syndict['streams_into_node'][str(n)]
        data['streams_out_of_node'][n] = syndict['streams_out_of_node'][str(n)]
    for t in data['time_periods']:
        data['investment_limit'][t] = syndict['investment_limit'][str(t)]
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
                data['conversion_rate'][(p,m,t)] = syndict['eta'][str((p,m,t))]
                data['capacity_limit'][(p,m,t)] = syndict['cap'][str((p,m,t))]
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

    # TODO: You need to find variable bounds or do something about the bigm
    # stuff in the gams file!
    TransformationFactory('gdp.bigm').apply_to(m, bigM=1e7)
    results = SolverFactory('baron').solve(m, tee=True)
    set_trace()




 # @model.Disjunction(model.T)
    # def Z1(m, t):
    #     return [[m.X[4,t] <= log(1 + m.X[2,t])], [m.X[2,t] <= 0, m.X[4,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z2(m, t):
    #     return [[m.X[5,t] <= 1.2*log(1 + m.X[3,t])], 
    #             [m.X[3,t] <= 0,m.X[5,t] <= 0] ]
    
    # @model.Disjunction(model.T)
    # def Z3(m, t):
    #     return [[m.X[13,t] == 0.75*m.X[9,t]], [m.X[9,t] <= 0, m.X[13,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z4(m, t):
    #     return [[m.X[14,t] <= 1.5*log(1 + m.X[10,t])], 
    #             [m.X[10,t] <= 0, m.X[14,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z5(m, t):
    #     return [[m.X[15,t] == m.X[11,t], m.X[15,t] == 0.5*m.X[12,t]], 
    #             [m.X[11,t] <= 0, m.X[12,t] <= 0, m.X[15,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z6(m, t):
    #     return [[m.X[21,t] <= 1.25*log(1 + m.X[16,t])], 
    #             [m.X[16,t] <= 0, m.X[21,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z7(m, t):
    #     return [[m.X[22,t] <= 0.9*log(1 + m.X[17,t])], 
    #             [m.X[17,t] <= 0, m.X[22,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z8(m, t):
    #     return [[m.X[23,t] <= log(1 + m.X[14,t])], 
    #             [m.X[23,t] <= 0, m.X[14,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z9(m, t):
    #     return [[m.X[24,t] == 0.9*m.X[18,t]], 
    #             [m.X[18,t] <= 0, m.X[24,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z10(m, t):
    #     return [[m.X[25,t] == 0.6*m.X[19,t]], 
    #             [m.X[19,t] <= 0, m.X[25,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z11(m, t):
    #     return [[m.X[26,t] <= 1.1*log(1 + m.X[20,t])], 
    #             [m.X[20,t] <= 0, m.X[26,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z12(m, t):
    #     return [[m.X[37,t] == 0.9*m.X[21,t], m.X[37,t] == m.X[39,t]], 
    #             [m.X[21,t] <= 0, m.X[29,t] <= 0, m.X[37,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z13(m, t):
    #     return [[m.X[38,t] <= log(1 + m.X[22,t])], 
    #             [m.X[22,t] <= 0, m.X[38,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z14(m, t):
    #     return [[m.X[39,t] <= 0.7*log(1 + m.X[27,t])], 
    #             [m.X[27,t] <= 0, m.X[39,t] <= 0]]
    
    # @model.Disjunction(model.T)
    # def Z15(m, t):
    #     return [[m.X[40,t] <= 0.65*log(1 + m.X[28,t]), 
    #              m.X[40,t] <= 0.65*log(1 + m.X[31,t])], 
    #             [m.X[28,t] <= 0, m.X[31,t] <= 0, m.X[40,t] <= 0]]
    
    # @model.Disjunction(model.T)
    # def Z16(m, t):
    #     return [[m.X[41,t] == m.X[32,t]], 
    #             [m.X[32,t] <= 0, m.X[41,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z17(m, t):
    #     return [[m.X[42,t] == m.X[33,t]], 
    #             [m.X[33,t] <= 0, m.X[42,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z18(m, t):
    #     return [[m.X[43,t] <= 0.75*log(1 + m.X[34,t])], 
    #             [m.X[34,t] <= 0, m.X[43,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z19(m, t):
    #     return [[m.X[44,t] <= 0.8*log(1 + m.X[35,t])], 
    #             [m.X[35,t] <= 0, m.X[44,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z20(m, t):
    #     return [[m.X[45,t] <= 0.85*log(1 + m.X[36,t])], 
    #             [m.X[36,t] <= 0, m.X[45,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z21(m, t):
    #     return [[m.X[49,t] <= log(1 + m.X[47,t])], 
    #             [m.X[47,t] <= 0, m.X[49,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z22(m, t):
    #     return [[m.X[50,t] <= 1.2*log(1 + m.X[48,t])], 
    #             [m.X[48,t] <= 0, m.X[50,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z23(m, t):
    #     return [[m.X[58,t] == 0.75*m.X[54,t]], 
    #             [m.X[58,t] <= 0, m.X[54,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z24(m, t):
    #     return [[m.X[59,t] <= 1.5*log(1 + m.X[55,t])], 
    #             [m.X[55,t] <= 0, m.X[59,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z25(m, t):
    #     return [[m.X[60,t] == m.X[56,t], m.X[60,t] == 0.5*m.X[57,t]], 
    #             [m.X[56,t] <= 0, m.X[57,t] <= 0, m.X[60,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z26(m, t):
    #     return [[m.X[66,t] <= 1.25*log(1 + m.X[61,t])], 
    #             [m.X[61,t] <= 0, m.X[66,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z27(m, t):
    #     return [[m.X[67,t] <= 0.9*log(1 + m.X[62,t])], 
    #             [m.X[62,t] <= 0, m.X[67,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z28(m, t):
    #     return [[m.X[68,t] <= log(1 + m.X[59,t])], 
    #             [m.X[59,t] <= 0, m.X[68,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z29(m, t):
    #     return [[m.X[69,t] == 0.9*m.X[63,t]], 
    #             [m.X[63,t] <= 0, m.X[69,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z30(m, t):
    #     return [[m.X[70,t] == 0.6*m.X[64,t]], 
    #             [m.X[64,t] <= 0, m.X[70,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z31(m, t):
    #     return [[m.X[71,t] <= 1.1*log(1 + m.X[65,t])], 
    #             [m.X[65,t] <= 0, m.X[71,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z32(m, t):
    #     return [[m.X[82,t] == 0.9*m.X[66,t], m.X[82,t] == m.X[74,t]], 
    #             [m.X[66,t] <= 0, m.X[74,t] <= 0, m.X[82,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z33(m, t):
    #     return [[m.X[83,t] <= log(1 + m.X[67,t])], 
    #             [m.X[67,t] <= 0, m.X[83,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z34(m, t):
    #     return [[m.X[84,t] <= 0.7*log(1 + m.X[72,t])], 
    #             [m.X[72,t] <= 0, m.X[84,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z35(m, t):
    #     return [[m.X[85,t] <= 0.65*log(1 + m.X[73,t]), 
    #              m.X[85,t] <= 0.65*log(1 + m.X[76,t])], 
    #             [m.X[73,t] <= 0, m.X[76,t] <= 0, m.X[85,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z36(m, t):
    #     return [[m.X[86,t] == m.X[77,t]], 
    #             [m.X[77,t] <= 0, m.X[86,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z37(m, t):
    #     return [[m.X[87,t] == m.X[78,t]], 
    #             [m.X[78,t] <= 0, m.X[87,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z38(m, t):
    #     return [[m.X[88,t] <= 0.75*log(1 + m.X[79,t])], 
    #             [m.X[79,t] <= 0, m.X[88,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z39(m, t):
    #     return [[m.X[89,t] <= 0.8*log(1 + m.X[80,t])], 
    #             [m.X[80,t] <= 0, m.X[89,t] <= 0]]

    # @model.Disjunction(model.T)
    # def Z40(m, t):
    #     return [[m.X[90,t] <= 0.85*log(1 + m.X[81,t])], 
    #             [m.X[81,t] <= 0, m.X[90,t] <= 0]]
