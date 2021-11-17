# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import random
import numpy as np
import sys

global distribution_dict_values
distribution_dict_values = {'MC': [], 'MMR': [], 'GPS ANT': [], 'LOC ANT Swi': [],
                            'GS ANT': [], 'LOC ANT': [], 'RA': [], 'RA ANT': [], 'NAV-4000': [],
                            'VOR ANT': [], 'MB ANT': [], 'ADF ANT': [], 'DME INT': [], 'ANT-42': []}
global n
component_distribution_dict = {
    'MC': {
        "name": "exponential",
        # "distribution":,
        "lambada": 1/12000,
        "mu": None,
        "sigma": None,
        "beta": None,
        "eta": None
        },
    'MMR': {
        "name": "exponential",
        # "distribution":,
        "lambada": 1/26000,
        "mu": None,
        "sigma": None,
        "beta": None,
        "eta": None
    },
    'GPS ANT': {
        "name": "weibull",
        # "distribution": weibull_distribution,
        "lambada": None,
        "mu": None,
        "sigma": None,
        "beta": 0.98,
        "eta": 26213
    },
    'LOC ANT Swi': {
        "name": "log_normal",
        # "distribution": gumbel_distribution,
        "lambada": None,
        "mu": 9.86,
        "sigma": 1.31,
        "beta": None,
        "eta": None
    },
    'GS ANT': {
        "name": "weibull",
        "lambada": None,
        "mu": None,
        "sigma": None,
        "beta": 1.01,
        "eta": 25326
    },
    'LOC ANT': {
        "name": "weibull",
        "lambada": None,
        "mu": None,
        "sigma": None,
        "beta": 0.86,
        "eta": 31636
    },
    'RA': {
        "name": "exponential",
        # "distribution":,
        "lambada": 1/80000,
        "mu": None,
        "sigma": None,
        "beta": None,
        "eta": None
    },
    'RA ANT': {
        "name": "weibull",
        "lambada": None,
        "mu": None,
        "sigma": None,
        "beta": 1.23,
        "eta": 35380
    },
    'NAV-4000': {
        "name": "exponential",
        # "distribution":,
        "lambada": 1/20000,
        "mu": None,
        "sigma": None,
        "beta": None,
        "eta": None
    },
    'VOR ANT': {
        "name": "weibull",
        "lambada": None,
        "mu": None,
        "sigma": None,
        "beta": 1.15,
        "eta": 28263
    },
    'MB ANT': {
        "name": "weibull",
        "lambada": None,
        "mu": None,
        "sigma": None,
        "beta": 0.92,
        "eta": 24926
    },
    'ADF ANT': {
        "name": "weibull",
        "lambada": None,
        "mu": None,
        "sigma": None,
        "beta": 0.99,
        "eta": 21042
    },
    'DME INT': {
        "name": "exponential",
        # "distribution":,
        "lambada": 1 / 50000,
        "mu": None,
        "sigma": None,
        "beta": None,
        "eta": None
    },
    'ANT-42': {
        "name": "weibull",
        "lambada": None,
        "mu": None,
        "sigma": None,
        "beta": 0.88,
        "eta": 51656
    }
}

'''
    this func create a seed
'''
def set_seed():
    seed_value = random.randrange(sys.maxsize)
    return seed_value

'''
uniform distribution
'''
def uniform_dis(seed_value):
    random.seed(seed_value)
    num = np.random.uniform(0, 1, None)
    return num
''' 
    exponential
'''
def exp_dis(uniform_values,lambada):
    nums = []
    for i in range(len(uniform_values)-1):
        nums.append(-math.log(uniform_values[i], math.e) / lambada)
    return nums

''' 
    weibull
'''
def weibull_dis(uniform_values, beta, eta): #eta=k?
    nums = []
    for i in range(len(uniform_values)-1):
        nums.append(eta * (math.pow(-math.log(uniform_values[i], math.e), 1 / beta)))
    return nums

def log_normal_dis(uniform_values,sigma,mu):
    nums = []
    for i in range(len(uniform_values)-1):
        nums.append(((math.sqrt(-2*math.log(uniform_values[i], math.e)))*math.sin(2*math.pi*uniform_values[i+1])*sigma)+mu)
    return nums

def create_random_array_by_uniform_dist(n):

    uniform_values = []
    for i in range(n):
            # uniform_value = uniform_dis(seed_value)
            uniform_value = uniform_dis(0.5)
            uniform_values.append(uniform_value)
    return uniform_values
'''
    create array of numbers for every sub system -
    this array choose min value between systems
'''
def compare_estimate():
    for sub_sys in sub_sys_values.keys():
        eval(sub_sys + "()")

def MC():
    MC = randoms_array_for_sys(['MC'])
    return MC

def MMR():
    MMR1 = randoms_array_for_sys(['MMR','GPS ANT','LOC ANT Swi'])
    MMR1 = create_min_array(MMR1)
    MMR2 = randoms_array_for_sys(['MMR','GPS ANT','LOC ANT Swi'])
    MMR3= randoms_array_for_sys(['GS ANT', 'LOC ANT'])


'''
    create randoms arrays
    get: list of numbers for systems 
    return one list of random values with minimum for every index
'''
def randoms_array_for_sys(list):
    n = 500
    for sys in list:

        print(sys)
        uniform_values = create_random_array_by_uniform_dist(n + 1)
        if component_distribution_dict[sys]["name"] == "exponential":
            lambada = component_distribution_dict[sys]["lambada"]
            values = exp_dis(uniform_values, lambada)


        elif component_distribution_dict[sys]["name"] == "weibull":
            beta = component_distribution_dict[sys]["beta"]
            eta = component_distribution_dict[sys]["eta"]
            values = weibull_dis(uniform_values,beta,eta)


        elif component_distribution_dict[sys]["name"] == "log_normal":
            mu = component_distribution_dict[sys]["mu"]
            sigma = component_distribution_dict[sys]["sigma"]
            values = log_normal_dis(uniform_values,sigma,mu)


        distribution_dict_values[sys]=values
    return distribution_dict_values



'''
    choose min value between arrays
'''
# def create_min_array(dict_of_sys_values):
#     min_array = []
#     for i in range(n):





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    seed_value = set_seed()

    # distribution_dict_values = {'MC': [], 'MMR': [], 'GPS ANT': [], 'LOC ANT Swi': [],
    # 'GS ANT': [], 'LOC ANT': [], 'RA': [], 'RA ANT':[], 'NAV-4000': [],
    # 'VOR ANT': [], 'MB ANT': [], 'ADF ANT': [], 'DME INT': [], 'ANT-42': []}

    sub_sys_values = {'MC': ['MC'], 'MMR': ['MMR','GPS ANT','LOC ANT Swi', 'GS ANT', 'LOC ANT'], 'RA': ['RA', 'RA ANT'],
                      'VHF-NAV': ['NAV-4000', 'VOR ANT', 'MB ANT','ADF ANT'], 'DME': ['DME INT', 'ANT-42']}

    n = 500

    for distribution in component_distribution_dict.keys():

        print(distribution)
        uniform_values = create_random_array_by_uniform_dist(n + 1)
        if component_distribution_dict[distribution]["name"] == "exponential":
            lambada = component_distribution_dict[distribution]["lambada"]
            values = exp_dis(uniform_values, lambada)


        elif component_distribution_dict[distribution]["name"] == "weibull":
            beta = component_distribution_dict[distribution]["beta"]
            eta = component_distribution_dict[distribution]["eta"]
            values = weibull_dis(uniform_values,beta,eta)


        elif component_distribution_dict[distribution]["name"] == "log_normal":
            mu = component_distribution_dict[distribution]["mu"]
            sigma = component_distribution_dict[distribution]["sigma"]
            values = log_normal_dis(uniform_values,sigma,mu)


        distribution_dict_values[distribution]=values

    all_sys = {}
    median_dict={}
    for distribution in distribution_dict_values:
        median = np.median(distribution_dict_values[distribution])
        median_dict[distribution] = median
        sum_dis = np.sum(distribution_dict_values[distribution])
        num = len(distribution_dict_values[distribution])
        all_sys[distribution] = (sum_dis) / (num)

    compare_estimate()








# See PyCharm help at https://www.jetbrains.com/help/pycharm/
