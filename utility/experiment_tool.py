from utility.setting import *
from optimization_methods.compare_method.simulated_annealing import *
from optimization_methods.compare_method.trust_region_method import *
from optimization_methods.compare_method.simple_mathmatical_optimization import *
from optimization_methods.compare_method.bayse_optimization import *
from optimization_methods.compare_method.select_best_data import *
from optimization_methods.proposal_method.frank_wolfe import FrankWolfe
from optimization_methods.proposal_method.frank_wolfe2 import FrankWolfe2
from optimization_methods.proposal_method.step_distance import StepDistance


def check_constraint():
    # if (not g.select_opt == FRANKWOLFE ) and (not g.n_nearest == n_nearest_list[0]):
    #     return False 
    if not g.select_data_type in constr_problem_data_for_opt[g.select_problem]:
        return False
    if not g.select_ml in constr_ml_for_opt[g.select_opt]:
        return False
    if g.n_feature < g.n_user_available_x:
        return False
    if g.n_user_available_x in bad_constr_problem_ufeature_for_opt[g.select_problem]:
        return False
    if g.n_feature in bad_constr_data_feature_for_opt[g.select_data_type]:
        return False
    if only_user_available_x and g.n_feature != g.n_user_available_x:
        return False
    if only_appropriate_feature and not g.n_feature in appropriate_data_feature_for_opt[g.select_data_type]:
        return False
    if auto_n_nearest and int(g.n_data * n_nearest_best_rate) != g.n_nearest:
        return False
    return True

def select_opt_problem():
    if g.select_opt == SIMULATEDANNEALING:
        return SimulatedAnnealing()
    elif g.select_opt == TRUSTREGION:
        return TrustRegionMethod()
    elif g.select_opt == MATHMATICALOPTIMIZATION:
        return MathmaticalOptimization()
    elif g.select_opt == FRANKWOLFE:
        return FrankWolfe()
    elif g.select_opt == FRANKWOLFE2:
        return FrankWolfe2()
    elif g.select_opt == STEPDISTANCE:
        return StepDistance()
    elif g.select_opt == BAYESIANOPTIMIZATIONMU:
        return BayseOptimizationMu()
    elif g.select_opt == BAYESIANOPTIMIZATIONLCB:
        return BayseOptimizationLCB()
    elif g.select_opt == SELECTBESTDATA:
        return SelectBestData()