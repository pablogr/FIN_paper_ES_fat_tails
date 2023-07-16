'''
This module contains the functions to calculate ES for different sample sizes (i.e. different numbers of datapoints).
'''

from os import path
from gc import collect
from shutil import copyfile
from math import isnan
import numpy as np
from scipy.stats import norm, nct, genhyperbolic, levy_stable
import pandas as pd
from module_generic_functions import dict_ret

dict_df_fitting = {'norm': 'normal', 'nct': 'non-centered t-student', 'genhyperbolic': 'generalized hyperbolic', 'levy_stable': 'Levy stable'}
line0   = "  N,    <ES>,     stdev_ES,    CI_1,      median_ES,    CI_2"
line00  = " --------------- FROM DISCRETE DATAPOINTS ------------------- ;  ----------------- FROM CONTINUOUS DISTIBUTIONS ------------------  \n"
line000 = "  N,    <ES>,     stdev_ES,    CI_1,      median_ES,    CI_2;       N,    <ES>,     stdev_ES,      CI_1,     median_ES,    CI_2"

# -----------------------------------------------------------------------------------------------------------------------

def truncate_array(dataset_in, truncation_limit):
    ''' This function removes from dataset_in all the values below a threshold (truncation limit).

    :param dataset_in: (numpy array) The array to truncate. It must be sorted.
    :param truncation_limit: (float) Number such that all values below it are removed from the input array.
    :return: truncated dataset_in
    '''

    if (truncation_limit == -np.infty):
        return dataset_in

    i = 0
    while (dataset_in[i] <= truncation_limit):
        i += 1
        if (i == len(dataset_in)):
            raise Exception("\n ERROR: The truncation limit (" + str(truncation_limit) + ") made all the elements of the array to be discarded. Please, check your data.\n")

    return dataset_in[i:]

# ------------------------------------------------------------------------------------------------

def calc_ES_discrete(data_in, truncation_limit=-70, alpha=0.025):
    ''' This function calculates the Expected Shortfall using the EBA formula (see page 18 of
    Final Draft RTS on the calculation of the stress scenario risk measure under Article 325bk(3) of Regulation (EU)
    No 575/2013 (Capital Requirements Regulation 2 – CRR2), 17 Dec. 2020. (EBA/RTS/2020/12).
    Note that due to its mathematical definition, this formula returns positive numbers which correspond to losses.

    :param data_in: (numpy array of floats) input random variable whose ES will be calculated. IT MUST BE SORTED!
    :param truncation_limit: (float) Lower truncation limit of the input dataset.
    :param alpha: (float) lower probability range for the calculation of ES
    :return: expected_shortfall
    '''

    my_var = truncate_array(data_in, truncation_limit)

    alphaN = alpha * len(my_var)
    int_alphaN = int(alphaN)
    expected_shortfall = (sum(my_var[0:int_alphaN]) + ((alphaN - int_alphaN) * my_var[int_alphaN])) / (-alphaN)

    return expected_shortfall

# ------------------------------------------------------------------------------------------------

def calc_ES_continuous( fitted_time_series ):
    ''' This function returns the (numerical) integral of a symbolic function whose parameters are stored in the input
    parameter (fitted_time_series). Such symbolic function is either a truncated normal, nct, genhyperbolic or Levy stable function.
    Note that, again, we define ES to be positive (for losses).
    Explanations:
    integ_lim_1 below MUST NOT BE top_limit, because you only need to integrate the region with lowest 2.5% probability, no more.
    At the last line of this function we divide by alpha_for_ES, not by neo_alpha_trunc. This is again because our calculation covers 2.5% probability.

    :param fitted_time_series: (class FittedTimeSeries) object which includes the fitted time series of a dataset.
    :return: -integ (float): This is the expected shortfall of the dataset of the input object.
    '''

    # Initialization
    func_type = fitted_time_series.fitting_parameters['distribution_type']
    loc_in    = fitted_time_series.fitting_parameters['loc_param']
    scale_in  = fitted_time_series.fitting_parameters['scale_param']
    if (func_type=="norm"):
        skew_in = None; tail1_in=None; tail2_in=None
    elif (func_type=="nct"):
        skew_in  = fitted_time_series.fitting_parameters['skewness_param']
        tail1_in = fitted_time_series.fitting_parameters['df_param']
        tail2_in = None
    elif (func_type=="genhyperbolic"):
        skew_in  = fitted_time_series.fitting_parameters['b_param']
        tail1_in = fitted_time_series.fitting_parameters['a_param']
        tail2_in = fitted_time_series.fitting_parameters['p_param']
    elif (func_type=="levy_stable"):
        skew_in  = fitted_time_series.fitting_parameters['beta_param']
        tail1_in = fitted_time_series.fitting_parameters['alpha_param']
        tail2_in = None
    else:
        raise Exception("\nERROR: Unrecognized function type "+str(func_type)+".\n")

    N_points_for_integ = 1000000

    if (func_type == "norm"):
        myvar0 = norm(loc=loc_in, scale=scale_in )
    elif (func_type == "nct"):
        myvar0 = nct(loc=loc_in, scale=scale_in, nc=skew_in, df=tail1_in)
    elif (func_type == "genhyperbolic"):
        myvar0 = genhyperbolic(loc=loc_in, scale=scale_in, b=skew_in, a=tail1_in, p=tail2_in)
    elif (func_type == "levy_stable"):
        myvar0 = levy_stable( loc=loc_in, scale=scale_in, beta=skew_in, alpha=tail1_in)
    else:
        raise Exception("\nERROR: Unknown function type ("+str(func_type)+").\n")

    # We define the integration limits. Note that we must take into account that part of the probability of was discarded due to the truncation.
    integ_lim_0 = fitted_time_series.bottom_limit
    try:
        neo_alpha_trunc = fitted_time_series.alpha_for_ES + myvar0.cdf(fitted_time_series.bottom_limit)
    except:
        print("\r WARNING: Numerical error; ",fitted_time_series.fitting_parameters)
        return None
    integ_lim_1 = myvar0.ppf( neo_alpha_trunc)  # Remember << myvar =norm( loc=0, scale=1 );  myvar.ppf(0.975) >> gives: 1.95997. << myvar.cdf(-1.96) >> gives 0.024998. Hence ppf is the inverse of cdf. ppf returns a value of "x" (not of probability), this is if loc increases in 10, then the output of ppt also does.

    # OLD, DEPRECATED: renorm_factor = 1 / (1 - 2 * myvar0.cdf( fitted_time_series.truncation_limit))  # This accounts for the amount of probability which we have discarded through truncation. We assume that it is equally distributed throughout all the non-discarded points.
    term0 = myvar0.cdf(fitted_time_series.bottom_limit)
    term1 = myvar0.cdf(fitted_time_series.top_limit)

    if ( (term1<term0 ) or (isnan(term1) or (isnan(term0))) or (term1<0.8) ): # Overflow. This can happen e.g. with Nsample=20 and ghyp.
        den_renorm_factor = 1-2*term0
        if ( ( (den_renorm_factor)<0.95) or ( (den_renorm_factor)>1.0000000001) or isnan(den_renorm_factor)):
            print("WARNING: The amount of probability distribution between truncation limits is "+(str(term1-term0))+"; calculation aborted.\n")
            return None
        renorm_factor = 1 / den_renorm_factor
    else:
        renorm_factor = 1 / (  myvar0.cdf( fitted_time_series.top_limit) -  myvar0.cdf( fitted_time_series.bottom_limit ) )

    # Actual calculation of the (numerical) integral
    x_integ = np.linspace(integ_lim_0, integ_lim_1, N_points_for_integ)
    myvar = myvar0.pdf(x_integ)
    myvar = myvar * x_integ
    integ = sum(myvar) * ((integ_lim_1 - integ_lim_0) / len(myvar))
    integ -= (myvar[0] + myvar[-1]) * ((integ_lim_1 - integ_lim_0) / (2 * len(myvar)))
    integ *= renorm_factor / fitted_time_series.alpha_for_ES


    ''' Old code con ejemplo de prueba q funciona:
    loc_in   = 0#xx fitted_time_series.fitting_param_loc
    scale_in = 0.05#xx fitted_time_series.fitting_param_scale
    skew_in  = 0 #xx  fitted_time_series.fitting_param_skew
    tail1_in = 2.5#xx  fitted_time_series.fitting_param_tails1
    tail2_in = 0 #xx fitted_time_series.fitting_param_tails2
    N_points_for_integ = Nin 

    if (func_type== "nct"):
       myvar0 = nct( loc=loc_in, scale=scale_in, nc=skew_in, df=tail1_in )
    elif (func_type== "genhyperbolic"):
       myvar0 = nct( loc=loc_in, scale=scale_in, b=skew_in, a=tail1_in, p=tail2_in )

    # We define the integration limits. Note that we must take into account that part of the probability of was discarded due to the truncation.
    integ_lim_0 = -0.7 #xx fitted_time_series.truncation_limit
    neo_alpha_trunc = 0.025 + myvar0.cdf( integ_lim_0 )  # xx fitted_time_series.alpha_for_ES + myvar0.cdf( fitted_time_series.truncation_limit)
    integ_lim_1 = myvar0.ppf( neo_alpha_trunc )          # Remember << myvar =norm( loc=0, scale=1 );  myvar.ppf(0.975) >> gives: 1.95997. << myvar.cdf(-1.96) >> gives 0.024998. Hence ppf is the inverse of cdf. ppf returns a value of "x" (not of probability), this is if loc increases in 10, then the output of ppt also does.
    renorm_factor = 1 / (1 - 2 * myvar0.cdf(-0.7))  # xx 0.7!! This accounts for the amount of probability which we have discarded through truncation. We assume that it is equally distributed throughout all the non-discarded points.

    # Actual calculation of the (numerical) integral
    x_integ = np.linspace(integ_lim_0, integ_lim_1, N_points_for_integ)
    myvar = myvar0.pdf( x_integ )
    myvar = myvar * x_integ
    integ = sum(myvar)*( (integ_lim_1 - integ_lim_0)/len(myvar) )
    integ -= ( myvar[0] + myvar[-1] )*( (integ_lim_1 - integ_lim_0)/(2*len(myvar)) )
    integ *= renorm_factor / 0.025 # 

    '''
    #print(" Params in:",loc_in,scale_in,skew_in,tail1_in,tail2_in," -->",-integ)

    return -integ

# ------------------------------------------------------------------------------------------------

def create_file_ES_vs_N( input_params, calc_mode ):
    ''' This function initializes the file where the results will be stored.

    :param input_params: (class) Set of input parameters of the calculation.
    :return: filepath1, filepath2 (str, str). If calc_ES_HS, these two files are the files which store ES and Avg.
                                              If calc_ES_distrib, they store the ES_from_discrete_dataset and ES_from_continuous_distrib.
    '''


    from datetime import datetime

    if not ((input_params.calc_ES_HS) or (input_params.calc_ES_distrib) or (input_params.calculation_mode=="ES_distrib-vs-N_last_dates")): return
    if ((input_params.calc_ES_HS) and (calc_mode!="ES_HS-vs-N")): return None, None
    if ((input_params.calc_ES_distrib) and (calc_mode == "ES_HS-vs-N") ): return None, None
    col_names = []
    for prod_label in input_params.list_product_labels[1]:
        for distrib_type in input_params.list_distribution_types:
            for sample_size in input_params.array_N_values:
                col_label = str(prod_label)+"/"+ str(distrib_type) + "/" + str(input_params.type_of_return + "/size" + str(int(sample_size))  )
                col_names.append(col_label)

    if (input_params.calculation_mode=="ES_distrib-vs-N_last_dates"):
        df_aux = pd.read_csv( path.join(input_params.input_directory, "ret_"+input_params.list_product_labels[1][0]+".csv") )
        file_index = [i for i in range(len(df_aux)-1)]
        del df_aux
    else:
        file_index = [ i for i in range( input_params.n_random_trials_calcES) ]

    df0 = pd.DataFrame(index=file_index, columns=col_names)

    infix = dict_ret[input_params.type_of_return]
    if (input_params.calc_ES_HS):
        filepath1 = path.join(input_params.output_fit_dir, "ES_HS_" + infix + str(input_params.list_product_labels[0]) + ".csv")
        filepath2 = path.join(input_params.output_fit_dir,"Avg_HS_" + infix + str(input_params.list_product_labels[0]) + ".csv")
        if not (input_params.only_plots):
            for filepath in [filepath1, filepath2]:
                if (path.exists(filepath)): copyfile(filepath, filepath.replace(".csv", "-"+str(datetime.now())+".csv"))
            df0.to_csv(filepath1, index=True)
            copyfile(filepath1,filepath2)
    else: # input_params.calc_ES_distrib is True
        filepath1 = path.join(input_params.output_fit_dir, "ES_distrib_discr_" + infix + str(input_params.list_product_labels[0]) + ".csv")
        filepath2 = path.join(input_params.output_fit_dir, "ES_distrib_cont_" + infix + str(input_params.list_product_labels[0]) + ".csv")
        if not (input_params.only_plots):
            for filepath in [filepath1, filepath2]:
                if (path.exists(filepath)): copyfile(filepath, filepath.replace(".csv", "-"+str(datetime.now())+".csv"))
            df0.to_csv(filepath1, index=True)
            copyfile(filepath1, filepath2)
    del df0

    return filepath1, filepath2

# -----------------------------------------------------------------------------------------------------------------------

def write_line_to_output_files( N, filepath_large, filepath_summary, array_results_in, col_label0, col_label, print_to_screen=0 ):
    '''This function stores the results. It adds one column to the large file which stores the raw output data, and adds one
    row (line) to the small file which stores a summary of the output data (i.e. mean, median, etc.).'''

    array_results = [ x for x in array_results_in if ( (x!=None) and (not isnan(x)) ) ]
    array_results = np.sort( array_results)
    siz = len(array_results)
    median_array = (array_results[round(np.floor((siz + 1) * 0.5) - 1)] + array_results[ round(np.ceil((siz + 1) * 0.5) - 1)]) / 2
    ci_1 = (array_results[round(np.floor((siz + 1) * 0.025) - 1)] + array_results[ round(np.ceil((siz + 1) * 0.025) - 1)]) / 2
    ci_2 = (array_results[round(np.floor((siz + 1) * 0.975) - 1)] + array_results[ round(np.ceil((siz + 1) * 0.975) - 1)]) / 2
    line_out_summary = "  " + str(N) + ', {:.8}'.format(np.mean(array_results)) + ', {:.8}'.format(  np.std(array_results)) + ', {:.8}'.format(ci_1) + ', {:.8}'.format(median_array) + ', {:.8}'.format(ci_2)
    if   (print_to_screen==1):  print(line_out_summary)
    elif (print_to_screen==-1): print(line_out_summary+"; ", end = '')
    f_summary = open(filepath_summary, 'a');
    f_summary.write(col_label0 + "," + line_out_summary.replace(" ", "") + "\n");
    f_summary.close()
    df_large = pd.read_csv(filepath_large, header=0)
    for i in range(len(df_large) - len(array_results_in)): array_results_in.append(None)
    df_large[col_label] = array_results_in
    df_large.to_csv(filepath_large, index=False)

    del f_summary; del df_large; del array_results; del array_results_in

    return

#------------------------------------------------------------------------------------------------------------------

def calc_random_array( distrib_type, N, ratio_aux, fp , top_limit, bottom_limit, verbose, k ):
    ''' This function generates an array of random number (synthetic data) from given parameters of a probability density
    function. The generated data lie within given truncation limits

    :param distrib_type: (str) Distribution type, e.g. "norm", "nct", "genhyperbolic", "levy_stable"
    :param N: (int) number of points of the output array (i.e. size of the sample to be analized).
    :param ratio_aux: (float) number to determine the size of the auxiliary array which is generated to refill the generated array after having removed outliers
    :param fp: (dict) parameters of the distribution to be used in the generation of the synthetic data
    :param top_limit: (float) upper limit of the generated data
    :param bottom_limit: (float) lower limit of the generated data
    :param verbose: (int) determines how much information is printed to screen
    :param k: (int) iteration number
    :return: (numpy array of floats) array of synthetic data which corresponds to the chosen probability density function.
    '''

    rand_array = np.array([float("NaN")])
    while ( isnan(rand_array[0])): # We repeat the generation until we make sure that all the outliers were removed

        aux_size = max(10, int(float(N) / ratio_aux))
        if (distrib_type == "norm"):
            rand_array = norm.rvs(loc=fp['loc_param'], scale=fp['scale_param'], size=N)
            aux_array = norm.rvs(loc=fp['loc_param'], scale=fp['scale_param'], size=aux_size)
        if (distrib_type == "nct"):
            rand_array = nct.rvs(loc=fp['loc_param'], scale=fp['scale_param'], nc=fp['skewness_param'], df=fp['df_param'],size=N)
            aux_array = nct.rvs(loc=fp['loc_param'], scale=fp['scale_param'], nc=fp['skewness_param'], df=fp['df_param'],size=aux_size)
        if (distrib_type == "genhyperbolic"):
            rand_array = genhyperbolic.rvs(loc=fp['loc_param'], scale=fp['scale_param'], a=fp['a_param'], b=fp['b_param'],p=fp['p_param'], size=N)
            aux_array = genhyperbolic.rvs(loc=fp['loc_param'], scale=fp['scale_param'], a=fp['a_param'], b=fp['b_param'], p=fp['p_param'], size=aux_size)
        if (distrib_type == "levy_stable"):
            rand_array = levy_stable.rvs(loc=fp['loc_param'], scale=fp['scale_param'], alpha=fp['alpha_param'], beta=fp['beta_param'], size=N)
            aux_array = levy_stable.rvs(loc=fp['loc_param'], scale=fp['scale_param'], alpha=fp['alpha_param'], beta=fp['beta_param'], size=aux_size)

        # Removal of outliers (i.e. synthetic data which lie beyond the established truncation limits) of the array

        jmin = 0
        for i in range(N):
            if ( isnan(rand_array[0])):
                break
            if ((rand_array[i] > top_limit) or (rand_array[i] < bottom_limit)):
                for j in range(jmin, aux_size):
                    if ((aux_array[j] < top_limit) and (aux_array[j] > bottom_limit)):
                        # print("  Hemos cambiado el ",i,"del rand_array (",rand_array[i],") por el ",j,"de aux_array (",aux_array[j],").")
                        rand_array[i] = aux_array[j]
                        jmin = j + 1
                        if (jmin >= aux_size):
                            rand_array = np.array([float("NaN")])
                            if (verbose>1): print("WARNING: The loop for refilling the array of random numbers after removal of outliers had to be repeated ("+str(distrib_type)+", "+str(N)+"; "+str(k)+"-th trial).")
                            break
                        break

    del aux_array

    return np.sort(rand_array)

#------------------------------------------------------------------------------------------------------------------

def ES_calculations( input_params, calc_ES_mode ):
    ''' This function calculates the ES and the average of synthetic data for different sizes of the sample data array.
    The synthetic data are generated from the given parameters of probability density functions (e.g. normal, nct), which
    are read from a file.
    It stores the result to a file.

    :param input_params: (class InputParams) object which contains the input information from the user
    :param calc_ES_mode: (str) Either "ES_HS-vs-N" or "ES_distrib-vs-N". In the former case it calculates just the ES
                                (and the average) of synthetic data using the historical simulation method (i.e. using
                                discrete data); in the later case, it fits to distributions and calculates the ES from both discrete data and continuous functions.
    :param input_params: (class InputParams)
    :return:
    '''

    from module_fitting import FittedTimeSeries
    from module_generic_functions import read_params_from_file
    from module_plots import plot_vs_N, plot_bare_histograms,  plot_discr_cont_vs_N

    if (input_params.calculation_mode == "ES_distrib-vs-N_last_dates"): return
    if ( not ( (input_params.calc_ES_HS and calc_ES_mode=="ES_HS-vs-N") or (input_params.calc_ES_distrib and calc_ES_mode=="ES_distrib-vs-N") ) or (input_params.calc_ES_HS=="ES_distrib-vs-N_last_dates") ): return
    assert calc_ES_mode in [None, "ES_HS-vs-N", "ES_distrib-vs-N"]
    print("Now running ES_calculations.")

    if (calc_ES_mode=="ES_HS-vs-N"):
        filepath_ES, filepath_Avg = create_file_ES_vs_N(input_params, calc_ES_mode)
        period_collect = 500
    else: # (calc_ES_mode=="ES_distrib-vs-N")
        input_params = update_input_parameters(input_params)
        filepath_ES_discr, filepath_ES_cont = create_file_ES_vs_N(input_params,calc_ES_mode)
        period_collect = 1

    top_limit         = input_params.top_limit
    bottom_limit      = input_params.bottom_limit

    if (calc_ES_mode=="ES_HS-vs-N"):
        line1 = "  N,    <Avg>,    stdev_Avg,     CI_1,       median_Avg,   CI_2"
        filepath_summary_1 = filepath_ES.replace( ".csv","_summary.csv")
        filepath_summary_2 = filepath_Avg.replace(".csv", "_summary.csv")
        if not input_params.only_plots:
            f     = open( filepath_summary_1,  'w'); f.write(     "product/distrib/ret,"+line0.replace(" ", "")+"\n"); f.close()
            f_avg = open( filepath_summary_2, 'w'); f_avg.write( "product/distrib/ret,"+line1.replace(" ", "")+"\n"); f_avg.close()
    else:  # (calc_ES_mode=="ES_distrib-vs-N")
        filepath_summary_1 = filepath_ES_discr.replace(".csv", "_summary.csv")
        filepath_summary_2 = filepath_ES_cont.replace(".csv", "_summary.csv")
        if not input_params.only_plots:
            f = open(filepath_summary_1, 'w'); f.write("product/distrib/ret," + line0.replace(" ", "") + "\n"); f.close()
            copyfile(filepath_summary_1,filepath_summary_2)


    if not input_params.only_plots:

        for prod_label in input_params.list_product_labels[1]:
            for distrib_type in input_params.list_distribution_types:

                fp = read_params_from_file( prod_label, distrib_type, input_params.type_of_return, input_params.output_fit_dir )

                if (calc_ES_mode=="ES_HS-vs-N"):
                    print("\n * Now calculating the historical ES for " + str(prod_label) + ", " + dict_df_fitting[str(distrib_type)]+" distribution:\n\n", line0)
                else:  # calc_ES_mode=="ES_distrib-vs-N"
                    print("\n * Now calculating the historical ES for " + str(prod_label) + ", " + dict_df_fitting[str(distrib_type)]+" distribution:\n\n", line00, line000)

                for N in input_params.array_N_values:

                    col_label0 = str(prod_label) + "/" + str(distrib_type) + "/" + str(input_params.type_of_return);
                    col_label = col_label0 + "/size" + str(int(N))
                    ES_results = [];
                    if (input_params.calc_ES_HS):
                        Avg_results = []
                    else: # input_params.calc_ES_distrib
                        ES_results_cont = []

                    for k in range(input_params.n_random_trials_calcES):

                        if input_params.calc_ES_distrib: print("\r  Now running calculation "+str(k+1)+" out of "+str(input_params.n_random_trials_calcES),end="")

                        # Generation of the array of synthetic data
                        rand_array = calc_random_array( distrib_type, N, input_params.ratio_aux, fp , top_limit, bottom_limit, input_params.verbose, k )

                        # Calculation of ES
                        ES_results.append(  calc_ES_discrete( rand_array ) )
                        if (calc_ES_mode=="ES_HS-vs-N"):
                            Avg_results.append( np.mean(rand_array) )
                        else: # calc_ES_mode=="ES_distrib-vs-N"
                            my_fitted_ts = FittedTimeSeries(input_params, prod_label, distrib_type, rand_array)
                            my_fitted_ts.fit_to_distribution()
                            ES_results_cont.append( calc_ES_continuous( my_fitted_ts ))

                        if ((k%period_collect)==0): collect()

                    # Saving results
                    if (calc_ES_mode=="ES_HS-vs-N"):
                        write_line_to_output_files( N, filepath_ES,  filepath_summary_1, ES_results, col_label0, col_label, 1)
                        write_line_to_output_files( N, filepath_Avg, filepath_summary_2, Avg_results, col_label0, col_label, 0 )
                    else:  # calc_ES_mode=="ES_distrib-vs-N"
                        print()
                        write_line_to_output_files(N, filepath_ES_discr, filepath_summary_1, ES_results, col_label0, col_label, -1)
                        write_line_to_output_files(N, filepath_ES_cont,  filepath_summary_2, ES_results_cont, col_label0, col_label, 1)

                    del ES_results
                    if (calc_ES_mode == "ES_HS-vs-N"): del Avg_results
                    else: del ES_results_cont

                    collect()

        del fp; del f


    if (input_params.make_plots):

        if (calc_ES_mode=="ES_HS-vs-N"):
            plot_vs_N( "ES",  input_params.output_directory, filepath_summary_1, "ES_vs_N_HS")
            plot_vs_N( "Avg", input_params.output_directory, filepath_summary_2, "Avg_vs_N_HS")
            plot_bare_histograms(input_params, filepath_ES, "ES")
            plot_bare_histograms(input_params, filepath_Avg, "Avg")
        else: # (calc_ES_mode=="ES_distrib-vs-N")
            plot_discr_cont_vs_N(input_params, "ES", filepath_summary_1, filepath_summary_2, "ES_vs_N_distrib_discr", "ES_vs_N_distrib_cont")
            plot_vs_N("ES", input_params.output_directory, filepath_summary_1, "ES_vs_N_distrib_discr" )
            plot_vs_N("ES", input_params.output_directory, filepath_summary_2, "ES_vs_N_distrib_cont" )
            plot_bare_histograms(input_params, filepath_ES_discr, "ES")
            plot_bare_histograms(input_params, filepath_ES_cont,  "ES")

    return

#------------------------------------------------------------------------------------------------------------------

def ES_calculations_last_dates(input_params):
    ''' This function calculates the ES and the average of synthetic data for different sizes of the sample data array, yet
    with the returns of the latest dates. The "calc_ES_mode" is "ES_distrib-vs-N".

    :param input_params: (class InputParams) object which contains the input information from the user
    :return:
    '''

    from module_fitting import FittedTimeSeries
    from module_plots import plot_vs_N, plot_bare_histograms, plot_discr_cont_vs_N

    print(" * Now running function ES_calculations_last_dates.")

    if not (  (input_params.calculation_mode in ["ES_distrib-vs-N_last_dates",None])): return

    input_params = update_input_parameters(input_params)
    filepath_ES_discr, filepath_ES_cont = create_file_ES_vs_N(input_params,"ES_distrib-vs-N_last_dates")
    period_collect = 1
    verbose = input_params.verbose

    filepath_summary_1 = filepath_ES_discr.replace(".csv", "_summary.csv")
    filepath_summary_2 = filepath_ES_cont.replace(".csv", "_summary.csv")
    if not input_params.only_plots:
        f = open(filepath_summary_1, 'w');
        f.write("product/distrib/ret," + line0.replace(" ", "") + "\n");
        f.close()
        copyfile(filepath_summary_1, filepath_summary_2)

    if not input_params.only_plots:

        for prod_label in input_params.list_product_labels[1]:
            for distrib_type in input_params.list_distribution_types:

                #fp = read_params_from_file(prod_label, distrib_type, input_params.type_of_return, input_params.output_fit_dir)
                file_fit_path = path.join(input_params.output_directory, input_params.list_product_labels[0], "fit_last_dates.csv")
                my_fitted_ts_whole = FittedTimeSeries(input_params, prod_label, distrib_type, file_fit_path )
                df_ts = pd.DataFrame( my_fitted_ts_whole.df_ts )
                for keyw in ["Open", "High", "Low", "Close", "Price", "Yield" ]:
                    if (keyw in list(df_ts)):
                        df_ts = df_ts.drop([keyw], axis=1)

                print("\n * Now calculating the historical ES for " + str(prod_label) + ", " + dict_df_fitting[ str(distrib_type)] + " distribution, WITH LAST DATES:\n\n", line00, line000)

                for N in input_params.array_N_values:

                    if (verbose==0): print(" N=",N,"-------------------------------------")

                    col_label0 = str(prod_label) + "/" + str(distrib_type) + "/" + str(input_params.type_of_return);
                    col_label = col_label0 + "/size" + str(int(N))
                    ES_results = [];
                    ES_results_cont = []
                    dim_k = len(my_fitted_ts_whole.df_ts)-N

                    for k in range(dim_k):

                        #if input_params.calc_ES_distrib: print("\r  Now running calculation "+str(k+1)+" out of "+str(dim_k),end="")
                        if (verbose>0): print("\r  Now running calculation with N="+str(N) +", dates from "+ str(df_ts.index[k]) +" to "+str(df_ts.index[k+N-1]), end="")

                        # Generation of the array of data (returns of last dates)
                        rand_array = np.sort( np.array( df_ts.iloc[k:k+N,0] ) )

                        # Calculation of ES
                        ES_results.append(calc_ES_discrete(rand_array))
                        my_fitted_ts = FittedTimeSeries(input_params, prod_label, distrib_type, rand_array)
                        #print(rand_array);print("type",type(rand_array));print(my_fitted_ts.input_from_array)
                        my_fitted_ts.fit_to_distribution()
                        ES_results_cont.append(calc_ES_continuous(my_fitted_ts))

                        if ((k % period_collect) == 0): collect()

                    # Saving results
                    write_line_to_output_files(N, filepath_ES_discr, filepath_summary_1, ES_results, col_label0, col_label, -1)
                    write_line_to_output_files(N, filepath_ES_cont, filepath_summary_2, ES_results_cont, col_label0, col_label, 1)

                    del ES_results; del ES_results_cont

                    collect()
        del f

    if (input_params.make_plots):

        plot_discr_cont_vs_N(input_params, "ES", filepath_summary_1, filepath_summary_2, "ES_vs_N_distrib_discr", "ES_vs_N_distrib_cont")
        plot_vs_N("ES", input_params.output_directory, filepath_summary_1, "ES_vs_N_distrib_discr")
        plot_vs_N("ES", input_params.output_directory, filepath_summary_2, "ES_vs_N_distrib_cont")
        plot_bare_histograms(input_params, filepath_ES_discr, "ES")
        plot_bare_histograms(input_params, filepath_ES_cont, "ES")

    return


# ------------------------------------------------------------------------------------------------------------------


def calc_ES_whole_dataset( input_params, prod_label, distrib_type ):
    '''This function calculates the ES from continuous probability density function which is read from a file which stores the
    parameters of that distribution.'''

    from module_fitting import FittedTimeSeries

    my_fitted_ts = FittedTimeSeries(input_params, prod_label, distrib_type, "fitted_parameters_file")
    my_fitted_ts.fit_to_distribution()
    ES_cont_whole_dataset = calc_ES_continuous(my_fitted_ts)

    print(" - The ES_cont_whole_dataset is",ES_cont_whole_dataset)

    del input_params; del my_fitted_ts; del prod_label; del distrib_type

    return ES_cont_whole_dataset

#------------------------------------------------------------------------------------------------------------------

def update_input_parameters(input_params):
    '''This function rewrites some of the parameters stored in the input_params object. Lowering how demanding the calculations are
    makes it feasible to run the calculations in mode ES_distrib-vs-N'''

    import input

    input_params.array_N_values             = input.array_N_values_distrib
    input_params.n_random_trials_calcES     = input.n_random_trials_calcES_distrib
    input_params.n_random_trials_fit        = input.n_random_trials_fit_distrib
    input_params.n_random_trials_fit_levy   = input.n_random_trials_fit_levy_distrib
    input_params.max_n_iter                 = input.max_n_iter_distrib
    input_params.max_n_iter_levy            = input.max_n_iter_levy_distrib

    return input_params

# ------------------------------------------------------------------------------------------------------------------
