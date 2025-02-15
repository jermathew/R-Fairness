"""
Provides some fairness measures definition
"""

import math
import numpy as np
import pandas as pd
import random

def exposure_avg(df):
    """Compute exposure avg"""
    list_position = df["position"].tolist()
    exp = 0
    for pos in list_position:
        exp += 1/math.log(pos+1, 2)
        #exp += random.randint(0, 100)

    exp_avg = exp / len(df)

    return exp_avg

def exposure_ratio(df):
    """Compute exposure avg"""
    list_position = df["position"].tolist()
    exp = 0
    for pos in list_position:
        exp += 1/math.log(pos+1, 2)

    num = exp

    den = 0
    for pos in range(len(list_position)):
        den += 1/math.log(pos+2, 2)

    exp_ratio = num/den

    return exp_ratio

def treatment_ratio(df):
    """Compute treatment ratio"""
    exp_ratio = exposure_ratio(df)

    list_utility = df["utility"].tolist()
    
    sum_utility = 0
    for utility in list_utility:
        sum_utility += utility

    mean_utility = sum_utility / len(list_utility)

    treatment_ratio = exp_ratio / mean_utility

    return treatment_ratio

def treatment_avg(df):
    """Compute treatment avg"""
    exp_ratio = exposure_avg(df)

    list_utility = df["utility"].tolist()
    
    sum_utility = 0
    for utility in list_utility:
        sum_utility += utility

    mean_utility = sum_utility / len(list_utility)

    treatment_ratio = exp_ratio / mean_utility

    return treatment_ratio

def kendall_tau_asymmetric(g1_df, g2_df):
    """Compute Kendall Tau distance"""
    assert not g1_df.empty and not g2_df.empty


    distance = 0
    normalisation_factor = 0.000000000001

    for i in range(len(g1_df)):
        for j in range(len(g2_df)):

            pos_i = g1_df.iloc[i].position
            pos_j = g2_df.iloc[j].position

            exp_i = 1/math.log(pos_i+1, 2)
            exp_j = 1/math.log(pos_j+1, 2)

            utility_i = g1_df.iloc[i].utility
            utility_j = g2_df.iloc[j].utility


            if np.sign((utility_i - utility_j) * (exp_i - exp_j)) < 0:
                inv = 1
            else:
                inv = 0
                
            distance += inv * (exp_i - exp_j) * abs(utility_i - utility_j)
            normalisation_factor += abs(exp_i - exp_j) * abs(utility_i - utility_j)

    distance = distance / normalisation_factor

    return distance

def demographic_parity(rank_list):
    """Compute demographic parity"""
    result = 0
    for rank in rank_list:
        result += 1/math.log(rank+1, 2)

    return result

def disparate_impact(rank_list, utility_list):
    """Compute Disparate impact"""
    assert len(rank_list) == len(utility_list)

    mean_utility = utility_list.mean()
    result = 0

    for rank, utility in zip(rank_list, utility_list):
        parity = 1/math.log(rank+1, 2)
        impact = parity * utility
        result += impact

    result = result / mean_utility
    return result

def kendall_tau(g1_df, g2_df):
    """Compute Kendall Tau distance"""
    assert not g1_df.empty and not g2_df.empty

    distance = 0
    normalisation_factor = 0

    for i in range(len(g1_df)):
        for j in range(len(g2_df)):

            pos_i = g1_df.iloc[i].position
            pos_j = g2_df.iloc[j].position

            exp_i = 1/math.log(pos_i+1, 2)
            exp_j = 1/math.log(pos_j+1, 2)

            utility_i = g1_df.iloc[i].utility
            utility_j = g2_df.iloc[j].utility

            distance += np.sign((utility_i - utility_j) * (exp_i - exp_j)) * abs(exp_i - exp_j) * abs(utility_i - utility_j)
            normalisation_factor += abs(exp_i - exp_j) * abs(utility_i - utility_j)

    distance = distance / normalisation_factor

    return distance

def kendall_tau_positive(g1_df, g2_df):
    """Compute Kendall Tau distance"""
    assert not g1_df.empty and not g2_df.empty

    distance = 0
    normalisation_factor = 0

    for i in range(len(g1_df)):
        for j in range(len(g2_df)):

            pos_i = g1_df.iloc[i].position
            pos_j = g2_df.iloc[j].position

            exp_i = 1/math.log(pos_i+1, 2)
            exp_j = 1/math.log(pos_j+1, 2)

            utility_i = g1_df.iloc[i].utility
            utility_j = g2_df.iloc[j].utility

            w = (utility_i - utility_j) * (exp_i - exp_j)

            inv = 0
            if  w < 0 and exp_i > exp_j:
                inv = -1
            else:
                inv = 0
                
            distance += inv * w
            normalisation_factor += abs(w) 

    distance = distance / normalisation_factor

    return distance

def kendall_tau_negative(g1_df, g2_df):
    """Compute Kendall Tau distance"""
    assert not g1_df.empty and not g2_df.empty

    distance = 0
    normalisation_factor = 0

    for i in range(len(g1_df)):
        for j in range(len(g2_df)):

            pos_i = g1_df.iloc[i].position
            pos_j = g2_df.iloc[j].position

            exp_i = 1/math.log(pos_i+1, 2)
            exp_j = 1/math.log(pos_j+1, 2)

            utility_i = g1_df.iloc[i].utility
            utility_j = g2_df.iloc[j].utility

            w = (utility_i - utility_j) * (exp_i - exp_j)

            inv = 0
            if  w < 0 and exp_i < exp_j:
                inv = +1
            else:
                inv = 0
                
            distance += inv * w
            normalisation_factor += abs(w) 

    distance = distance / normalisation_factor

    return distance


    