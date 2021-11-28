import math
import random
import sys
from scipy.stats import exponweib
from scipy.optimize import fmin
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

# import tensorflow_probability as tfp



class RNS_system:

    def __init__(self):
        self.distribution_dict_values = {'MC': [], 'MMR': [], 'GPS ANT': [], 'LOC ANT Swi': [],
                                    'GS ANT': [], 'LOC ANT': [], 'RA': [], 'RA ANT': [], 'NAV-4000': [],
                                    'VOR ANT': [], 'MB ANT': [], 'ADF ANT': [], 'DME INT': [], 'ANT-42': []}
        self.sub_sys_values = {'MC': ['MC'], 'MMR': ['MMR', 'GPS ANT', 'LOC ANT Swi', 'GS ANT', 'LOC ANT'],
                          'RA': ['RA', 'RA ANT'],
                          'VHF_NAV': ['NAV-4000', 'VOR ANT', 'MB ANT', 'ADF ANT'], 'DME': ['DME INT', 'ANT-42']}
        self.n = 500
        self.component_distribution_dict = {
            'MC': {
                "name": "exponential",
                # "distribution":,
                "lambada": 1 / 12000,
                "mu": None,
                "sigma": None,
                "beta": None,
                "eta": None
            },
            'MMR': {
                "name": "exponential",
                # "distribution":,
                "lambada": 1 / 26000,
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
                "lambada": 1 / 80000,
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
                "lambada": 1 / 20000,
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
        self.len_of_uniform_array = 501
        self.seed = 0.5
        self.esti_parameters = {'MC': [], 'MMR': [], 'GPS ANT': [], 'LOC ANT Swi': [],
                                    'GS ANT': [], 'LOC ANT': [], 'RA': [], 'RA ANT': [], 'NAV-4000': [],
                                    'VOR ANT': [], 'MB ANT': [], 'ADF ANT': [], 'DME INT': [], 'ANT-42': []}

        self.paramters_for_esti = {'MC_lambada':None, 'MMR_lambada':None, 'GPS ANT_beta':None, 'GPS ANT_eta':None, 'LOC ANT Swi_mu':None, 'LOC ANT Swi_sigma':None, 'GS ANT_beta':None, 'GS ANT_eta':None, 'LOC ANT_beta':None,'LOC ANT_eta':None, 'RA_lambada':None, 'RA ANT_beta':None,'RA ANT_eta':None, 'NAV-4000_lambada':None,
                                                 'VOR ANT_beta':None, 'VOR ANT_eta':None, 'MB ANT_beta':None,'MB ANT_eta':None, 'ADF ANT_beta':None, 'ADF ANT_eta':None, 'DME INT_lambada':None, 'ANT-42_beta':None, 'ANT-42_eta':None}
    '''
        set seed
    '''

    def set_seed(self):
        seed_value = random.randrange(sys.maxsize)
        self.seed = seed_value
        return seed_value

    '''
        create randoms arrays
        get: list of numbers for systems 
        return one list of random values 
    '''

    def randoms_array_for_sys(self, list_of_distrubition):
        sys_dict_values = {}
        for system in list_of_distrubition:
            uniform_values = self.create_random_array_by_uniform_dist()
            if self.component_distribution_dict[system]["name"] == "exponential":
                lambada = self.component_distribution_dict[system]["lambada"]
                values = self.exp_dis(uniform_values, lambada)
                # self.histograms(values, system, "expo")
                # estimate parameters
                esti_lambada = self.fitexp(values)
                self.paramters_for_esti[f'{system}_lambada'] = 1/esti_lambada
                self.esti_parameters[system] = esti_lambada


            elif self.component_distribution_dict[system]["name"] == "weibull":
                beta = self.component_distribution_dict[system]["beta"]
                eta = self.component_distribution_dict[system]["eta"]
                values = self.weibull_dis(uniform_values, beta, eta)
                # self.histograms(values, system, "weibull")
                # estimate parameters
                esti_beta, esti_eta = self.fitweibull(values)
                self.paramters_for_esti[f'{system}_beta'] = esti_beta
                self.paramters_for_esti[f'{system}_eta'] = esti_eta
                self.esti_parameters[system] = esti_beta, esti_eta



            elif self.component_distribution_dict[system]["name"] == "log_normal":
                mu = self.component_distribution_dict[system]["mu"]
                sigma = self.component_distribution_dict[system]["sigma"]
                values = self.log_normal_dis(uniform_values, sigma, mu)
                # self.histograms(values,system,"lognormal")
                #estimate parameters
                esti_mu, esti_sigma = self.fitlognormal(values)
                self.paramters_for_esti[f'{system}_mu'] = esti_mu
                self.paramters_for_esti[f'{system}_sigma'] = esti_sigma
                self.esti_parameters[system] = esti_mu, esti_sigma


            sys_dict_values[system] = values
        return sys_dict_values

    '''
        create random array for uniform distribution
    '''
    def create_random_array_by_uniform_dist(self):

        uniform_values = []
        for i in range(self.len_of_uniform_array):
            uniform_value = self.uniform_dis(self.seed)
            uniform_values.append(uniform_value)
        return uniform_values

    '''
    uniform distribution
    '''

    def uniform_dis(self, seed_value):
        random.seed(seed_value)
        num = np.random.uniform(0, 1, None)
        return num

    ''' 
        exponential
    '''

    def exp_dis(self,uniform_values, lambada):
        nums = []
        for i in range(len(uniform_values) - 1):
            nums.append(-math.log(uniform_values[i], math.e) / lambada)
        return nums

    ''' 
        weibull
    '''

    def weibull_dis(self,uniform_values, beta, eta):  # eta=k?
        nums = []
        for i in range(len(uniform_values) - 1):
            nums.append(eta * (math.pow(-math.log(uniform_values[i], math.e), 1 / beta)))
        return nums

    '''
        lognormal
    '''
    def log_normal_dis(self,uniform_values, sigma, mu):
        nums = []
        for i in range(len(uniform_values) - 1):
            uni_to_normal = math.sqrt(-2 * math.log(uniform_values[i], math.e)) * math.cos(2 * math.pi * uniform_values[i + 1])
            result = uni_to_normal * sigma + mu
            normal_to_lognormal = math.exp(result)
            nums.append(normal_to_lognormal)
        return nums

    def compare_estimate(self):
        sub_sys_after_compare = {}
        sub_sys_after_expectation = {}
        for sub_sys in self.sub_sys_values.keys():
            sys_values = eval('self.' + sub_sys + "()")
            sub_sys_after_compare[sub_sys] = sys_values
            ## calculate expectation
            E = self.calculate_expectation(sys_values)
            sub_sys_after_expectation[sub_sys] = E
        ## calculate expectation for whole system
        RNS_values = self.create_min_array(sub_sys_after_compare)
        E = self.calculate_expectation(RNS_values)
        sub_sys_after_expectation['RNS'] = E
        return sub_sys_after_expectation

    '''
    get list of values and calculate the expectation

    '''

    def calculate_expectation(self,values):
        length = len(values)
        summary = sum(values)
        expectation = summary / length
        return expectation

    '''
        choose min value between arrays
    '''

    def create_min_array(self,dict_of_sys_values):
        min_array = []
        for i in range(self.n):
            min_val = sys.maxsize
            for system in dict_of_sys_values.keys():
                curr_val = dict_of_sys_values[system][i]
                min_val = min(min_val, curr_val)
            min_array.append(min_val)
        return min_array

    '''
        choose min value between arrays
    '''

    def create_min_array(self, dict_of_sys_values):
        min_array = []
        for i in range(self.n):
            min_val = sys.maxsize
            for system in dict_of_sys_values.keys():
                curr_val = dict_of_sys_values[system][i]
                min_val = min(min_val, curr_val)
            min_array.append(min_val)
        return min_array

    ###### function for evry su system
    def MC(self):
        MC = self.randoms_array_for_sys(['MC'])['MC']
        return MC

    def MMR(self):
        # part a - 2 parallel process MMR1 and MMR2
        MMR1 = self.randoms_array_for_sys(['MMR', 'GPS ANT', 'LOC ANT Swi'])
        MMR1 = self.create_min_array(MMR1)

        MMR2 = self.randoms_array_for_sys(['MMR', 'GPS ANT', 'LOC ANT Swi'])
        MMR2 = self.create_min_array(MMR2)

        # find the maximum between the 2 process

        MMR_part_a = np.maximum(MMR1, MMR2)

        # part b end of process
        MMR3 = self.randoms_array_for_sys(['GS ANT', 'LOC ANT'])
        MMR3 = self.create_min_array(MMR3)

        # find the minimum between the part a and part b
        MMR_final = np.minimum(MMR_part_a, MMR3)

        return list(MMR_final)

    def RA(self):
        # part a - 2 parallel process RA1 and RA2
        RA1 = self.randoms_array_for_sys(['RA', 'RA ANT'])
        RA1 = self.create_min_array(RA1)
        # RA ANT happens twice
        RA_ANT = self.randoms_array_for_sys(['RA ANT'])['RA ANT']
        RA1 = np.minimum(RA1, RA_ANT)

        RA2 = self.randoms_array_for_sys(['RA', 'RA ANT'])
        RA2 = self.create_min_array(RA2)
        # RA ANT happens twice
        RA_ANT = self.randoms_array_for_sys(['RA ANT'])['RA ANT']
        RA2 = np.minimum(RA2, RA_ANT)

        # find the maximum between the 2 process
        RA_part_a = np.maximum(RA1, RA2)
        return list(RA_part_a)

    def VHF_NAV(self):

        # part a - 2 parallel process
        NAV4000_1 = self.randoms_array_for_sys(['NAV-4000'])['NAV-4000']
        NAV4000_2 = self.randoms_array_for_sys(['NAV-4000'])['NAV-4000']
        # find the maximum between the 2 process
        NAV4000 = np.maximum(NAV4000_1, NAV4000_2)

        # part b - one process
        VHF_NAV = self.randoms_array_for_sys(['VOR ANT', 'MB ANT', 'ADF ANT'])
        VHF_NAV = self.create_min_array(VHF_NAV)

        MMR_final = np.minimum(NAV4000, VHF_NAV)

        return list(MMR_final)

    def DME(self):

        # 2 parallel process
        DME1 = self.randoms_array_for_sys(['DME INT', 'ANT-42'])
        DME1 = self.create_min_array(DME1)

        DME2 = self.randoms_array_for_sys(['DME INT', 'ANT-42'])
        DME2 = self.create_min_array(DME2)
        DME_final = np.maximum(DME1, DME2)

        return list(DME_final)


    '''
    estimate weibull parameters
    # array_of_values is data array
    # returns [shape, scale]
    '''
    def fitweibull(self, array_of_values):
        def optfun(theta):
            return -np.sum(np.log(exponweib.pdf(array_of_values, 1, theta[0], scale=theta[1], loc=0)))

        logx = np.log(array_of_values)
        shape = 1.2 / np.std(logx)
        scale = np.exp(np.mean(logx) + (0.572 / shape))
        return fmin(optfun, [shape, scale], xtol=0.01, ftol=0.01, disp=0)

    '''
        estimate lognormal parameters
        # array_of_values is data array
        # returns [mu, sigma]
        '''
    def fitlognormal(self, array_of_values):
        # fit data
        sigma, loc, scale = stats.lognorm.fit(array_of_values, floc=0)

        # get mu
        mu = np.log(scale)
        return mu, sigma

    '''
            estimate exp parameters
            # array_of_values is data array
            # returns lambada
            '''

    def fitexp(self, array_of_values):
        # fit data
        sum_dis = np.sum(array_of_values)
        num = len(array_of_values)
        esti_lambada = (sum_dis) / (num)
        return esti_lambada

    '''
        Calculates halton function
    '''
    def Halton_function(self, count_of_number):
        # engine = qmc.Halton(d=2, seed=self.set_seed())
        # sample["Halton"] = engine.random(n_sample)
        list_of_uniform = []
        # for i in range(count_of_number):
        #     list_of_uniform.append(np.random.uniform(0, 1, None))
        # Xi = list_of_uniform.sample(count_of_number * 2, rule='halton')

        # tfp.mcmc.sample_halton_sequence(
        #     dim, num_results=None, sequence_indices=None, dtype=tf.float32, randomized=True,
        #     seed=None, name=None
        # )
        # result_df = generate_dataframe(sample_size, Xi)
        # print_result_values(result_df)
        print("a")


    def histograms(self,array,sys,dist):
        # rng = np.random.RandomState(10)  # deterministic random data
        # a = np.hstack((rng.normal(size=1000),
        #                rng.normal(loc=5, scale=2, size=1000)))
        _ = plt.hist(array, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram-"+sys+" "+dist)
        plt.show()


    '''
        create randoms arrays
        get: list of numbers for systems 
        return one list of random values 
    '''

    def randoms_array_for_sys_halton(self, list_of_distrubition,array_halton):
        sys_dict_values = {}
        for system in list_of_distrubition:
            # uniform_values = self.create_random_array_by_uniform_dist()
            if self.component_distribution_dict[system]["name"] == "exponential":
                lambada = self.component_distribution_dict[system]["lambada"]
                values = self.exp_dis(array_halton, lambada)
                # self.histograms(values, system, "expo")
                # estimate parameters
                esti_lambada = self.fitexp(values)
                self.esti_parameters[system] = esti_lambada


            elif self.component_distribution_dict[system]["name"] == "weibull":
                beta = self.component_distribution_dict[system]["beta"]
                eta = self.component_distribution_dict[system]["eta"]
                values = self.weibull_dis(array_halton, beta, eta)
                # self.histograms(values, system, "weibull")
                # estimate parameters
                esti_beta, esti_eta = self.fitweibull(values)
                self.esti_parameters[system] = esti_beta, esti_eta



            elif self.component_distribution_dict[system]["name"] == "log_normal":
                mu = self.component_distribution_dict[system]["mu"]
                sigma = self.component_distribution_dict[system]["sigma"]
                values = self.log_normal_dis(array_halton, sigma, mu)
                # self.histograms(values,system,"lognormal")
                #estimate parameters
                esti_mu, esti_sigma = self.fitlognormal(values)
                self.esti_parameters[system] = esti_mu, esti_sigma


            sys_dict_values[system] = values
        return sys_dict_values

    def confidence_interval(self):
        # expectation_100 = pd.DataFrame(columns=['MC', 'MMR', 'RA', 'VHF_NAV', 'DME', 'RNS'])
        expectation_100 = pd.DataFrame(columns=['MC_lambada', 'MMR_lambada', 'GPS ANT_beta', 'GPS ANT_eta', 'LOC ANT Swi_mu', 'LOC ANT Swi_sigma', 'GS ANT_beta', 'GS ANT_eta', 'LOC ANT_beta','LOC ANT_eta', 'RA_lambada', 'RA ANT_beta','RA ANT_eta', 'NAV-4000_lambada',
                                                 'VOR ANT_beta', 'VOR ANT_eta', 'MB ANT_beta','MB ANT_eta', 'ADF ANT_beta', 'ADF ANT_eta', 'DME INT_lambada', 'ANT-42_beta', 'ANT-42_eta'])
        number_of_trials = 100
        for i in range(number_of_trials):
            self.seed = self.set_seed()
            distribution_dict_values = self.randoms_array_for_sys(['MC', 'MMR', 'GPS ANT', 'LOC ANT Swi',
                                                                  'GS ANT', 'LOC ANT', 'RA', 'RA ANT', 'NAV-4000',
                                                                  'VOR ANT', 'MB ANT', 'ADF ANT', 'DME INT', 'ANT-42'])

            # curr_values = rns.compare_estimate()
            curr_values = self.paramters_for_esti
            # expectation_100.loc[len(expectation_100)] = list(curr_values.values())
            expectation_100=  expectation_100.append(curr_values, ignore_index=True)

        #  Confidence interval 90%
        bottom_decile = pd.DataFrame(columns=expectation_100.keys())
        top_decile = pd.DataFrame(columns=expectation_100.keys())
        for system in expectation_100.keys():
            expectation_100[system] = sorted(expectation_100[system])
            bottom_decile[system] = expectation_100[system].loc[0: 0.05 * number_of_trials - 1]
            top_decile[system] = expectation_100[system].loc[0.95 * number_of_trials: number_of_trials]

        merged_deciles = pd.concat([bottom_decile, top_decile])
        # (1) Confidence interval by Empirical experiments
        confidence_interval_empirical = stats.t.interval(0.9, len(merged_deciles) - 1, loc=np.mean(merged_deciles),
                                                         scale=stats.sem(merged_deciles))

        # (2) Confidence interval by normal distribution
        confidence_interval_normal = stats.t.interval(0.9, len(expectation_100) - 1, loc=np.mean(expectation_100),
                                                      scale=stats.sem(expectation_100))
        confidence_interval_result = pd.DataFrame(columns=expectation_100.columns)


        for i, system in enumerate(expectation_100.columns):
            [sys_col, param] = system.split("_")
            real_value = self.component_distribution_dict[sys_col][param]

            print(f"{system} value: {real_value}")
            print(f"confidence_interval_empirical_results:[ {confidence_interval_empirical[0][i]} : {confidence_interval_empirical[1][i]} ']")
            # print(f"confidence_interval_empirical_results max: {confidence_interval_empirical[1][i]}")
            print(f"confidence_interval_results:[ {confidence_interval_normal[0][i]} : {confidence_interval_normal[1][i]} ]")
            # print(f"confidence_interval_results max: {confidence_interval_normal[1][i]}")


            result = [True if confidence_interval_empirical[0][i] <= real_value <= confidence_interval_empirical[1][
                i] else False,
                   True if confidence_interval_normal[0][i] <= real_value <= confidence_interval_normal[1][i] else False]
            confidence_interval_result[system] = result

        return confidence_interval_result

