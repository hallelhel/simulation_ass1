# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
from scipy import stats
# from scipy.stats import qmc
import RNS_ASS
from statsmodels.tools import sequences


if __name__ == '__main__':

    ###############1A###################

    ## set seed
    rns = RNS_ASS.RNS_system()
    seed_value = rns.set_seed()

    ## lottery uniform 1 number
    uniform_number = rns.uniform_dis(seed_value)

    ## lottery list of uniform numbers
    uniform_numbers = rns.create_random_array_by_uniform_dist()

    ## lottery for every distribution depend on parameters
        ## lottery expotitonal number using uniform number
        ## lottery weibull number using uniform number
        ## lottery lognormal number using uniform number

    ###############1B###################
    # set seed to 0.5 and lottery 500 numbers for every 14 dist
    rns.seed = 0.5
    distribution_dict_values = rns.randoms_array_for_sys(['MC', 'MMR', 'GPS ANT', 'LOC ANT Swi',
                            'GS ANT', 'LOC ANT', 'RA', 'RA ANT', 'NAV-4000',
                            'VOR ANT', 'MB ANT', 'ADF ANT', 'DME INT', 'ANT-42'])

    distribution_estimates = rns.esti_parameters
    ## estimate parameters:
    # all_sys = {}
    # median_dict = {}
    # weibull_dict={}
    # for distribution in distribution_dict_values:
        # median = np.median(distribution_dict_values[distribution])
        # median_dict[distribution] = median
        # sum_dis = np.sum(distribution_dict_values[distribution])
        # num = len(distribution_dict_values[distribution])
        # all_sys[distribution] = (sum_dis) / (num)




    ### estimators to compare with table 3
    sub_sys_expectation1 = rns.compare_estimate()

    ###############1C###################
    # (1) lottery again 500 with seed=0.5
    rns.seed = 0.5

    distribution_dict_values = rns.randoms_array_for_sys(['MC', 'MMR', 'GPS ANT', 'LOC ANT Swi',
                                                          'GS ANT', 'LOC ANT', 'RA', 'RA ANT', 'NAV-4000',
                                                          'VOR ANT', 'MB ANT', 'ADF ANT', 'DME INT', 'ANT-42'])
    # sub_sys_expectation2 = rns.compare_estimate()
    distribution_estimates = rns.esti_parameters
    # (2) lottery again 500 with seed=othee value

    rns.seed = 2

    distribution_dict_values = rns.randoms_array_for_sys(['MC', 'MMR', 'GPS ANT', 'LOC ANT Swi',
                                                          'GS ANT', 'LOC ANT', 'RA', 'RA ANT', 'NAV-4000',
                                                          'VOR ANT', 'MB ANT', 'ADF ANT', 'DME INT', 'ANT-42'])
    # sub_sys_expectation3 = rns.compare_estimate()
    distribution_estimates = rns.esti_parameters


    # (3) lottery again now 10,000 with diff seed
    rns.seed = rns.set_seed()
    rns.n = 10000
    rns.len_of_uniform_array = 10001

    distribution_dict_values = rns.randoms_array_for_sys(['MC', 'MMR', 'GPS ANT', 'LOC ANT Swi',
                                                          'GS ANT', 'LOC ANT', 'RA', 'RA ANT', 'NAV-4000',
                                                          'VOR ANT', 'MB ANT', 'ADF ANT', 'DME INT', 'ANT-42'])
    # sub_sys_expectation4 = rns.compare_estimate()
    distribution_estimates = rns.esti_parameters
    ###############1D###################

    # 100 lottery every lottery diff seed
    rns.n = 500
    rns.len_of_uniform_array = 501
    expectation_100 = pd.DataFrame(columns=['MC', 'MMR', 'RA', 'VHF_NAV', 'DME', 'RNS'])
    number_of_trials = 100
    for i in range(number_of_trials):
        rns.seed = rns.set_seed()
        distribution_dict_values = rns.randoms_array_for_sys(['MC', 'MMR', 'GPS ANT', 'LOC ANT Swi',
                                                              'GS ANT', 'LOC ANT', 'RA', 'RA ANT', 'NAV-4000',
                                                             'VOR ANT', 'MB ANT', 'ADF ANT', 'DME INT', 'ANT-42'])
        curr_values = rns.compare_estimate()
        expectation_100.loc[len(expectation_100)] = list(curr_values.values())

    #  Confidence interval 90%
    bottom_decile = pd.DataFrame(columns=expectation_100.keys())
    top_decile = pd.DataFrame(columns=expectation_100.keys())
    for system in expectation_100.keys():
        expectation_100[system] = sorted(expectation_100[system])
        bottom_decile[system] = expectation_100[system].loc[0: 0.1 * number_of_trials - 1]
        top_decile[system] = expectation_100[system].loc[0.9 * number_of_trials: number_of_trials]

    merged_deciles = pd.concat([bottom_decile, top_decile])
    # (1) Confidence interval by Empirical experiments
    confidence_interval_empirical = stats.t.interval(0.9, len(merged_deciles) - 1, loc=np.mean(merged_deciles), scale= stats.sem(merged_deciles))

    # (2) Confidence interval by normal distribution
    confidence_interval_normal = stats.t.interval(0.9, len(expectation_100) - 1, loc=np.mean(expectation_100), scale= stats.sem(expectation_100))




    ###############1E###################
    ######## halton 50 #########
    halton_values = sequences.halton(dim=2, n_sample=50)
    halton_values_list = [val[0] for val in halton_values]
    distribution_dict_values = rns.randoms_array_for_sys_halton(['MC', 'MMR', 'GPS ANT', 'LOC ANT Swi',
                                                          'GS ANT', 'LOC ANT', 'RA', 'RA ANT', 'NAV-4000',
                                                          'VOR ANT', 'MB ANT', 'ADF ANT', 'DME INT', 'ANT-42'],
                                                       halton_values_list)
    distribution_estimates_50 = rns.esti_parameters
    # exp_50 = {}
    # for sys in distribution_dict_values.keys():
    #     exp_50[sys] = rns.calculate_expectation(distribution_dict_values[sys])
    data_items = distribution_estimates_50.items()
    data_list = list(data_items)
    df_50 = pd.DataFrame(data_list)

    df_50.to_csv('50_.csv')


    ######## halton 200 #########
    halton_values = sequences.halton(dim=1, n_sample=200)
    halton_values_list = [val[0] for val in halton_values]
    distribution_dict_values = rns.randoms_array_for_sys_halton(['MC', 'MMR', 'GPS ANT', 'LOC ANT Swi',
                                                          'GS ANT', 'LOC ANT', 'RA', 'RA ANT', 'NAV-4000',
                                                          'VOR ANT', 'MB ANT', 'ADF ANT', 'DME INT', 'ANT-42'],
                                                         halton_values_list)
    distribution_estimates_200 = rns.esti_parameters
    # exp_200 = {}
    # for sys in distribution_dict_values.keys():
    #     exp_200[sys] = rns.calculate_expectation(distribution_dict_values[sys])
    data_items = distribution_estimates_200.items()
    data_list = list(data_items)
    df_200 = pd.DataFrame(data_list)

    df_200.to_csv('200_.csv')

    ######## halton 500 #########
    halton_values = sequences.halton(dim=1, n_sample=500)
    halton_values_list = [val[0] for val in halton_values]
    distribution_dict_values = rns.randoms_array_for_sys_halton(['MC', 'MMR', 'GPS ANT', 'LOC ANT Swi',
                                                          'GS ANT', 'LOC ANT', 'RA', 'RA ANT', 'NAV-4000',
                                                          'VOR ANT', 'MB ANT', 'ADF ANT', 'DME INT', 'ANT-42'],
                                                         halton_values_list)
    distribution_estimates_500 = rns.esti_parameters
    data_items = distribution_estimates_500.items()
    data_list = list(data_items)
    df_500 = pd.DataFrame(data_list)

    df_500.to_csv('500_.csv')
    exp_500 = {}
    # for sys in distribution_dict_values.keys():
    #     exp_500[sys] = rns.calculate_expectation(distribution_dict_values[sys])

    ###############2A###################





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
