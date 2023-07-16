
import pandas as pd
import numpy as np
from os import path
from math import isnan


dict_ret = { "absolute":"abs_ret_", "relative":"rel_ret_", "logarithmic":"log_ret_" }

#------------------------------------------------------------------------------------------------

def print_message_2(country_name, n_years):
    print("\n ------------------------------------------------- \n       Now analysing " + str(
        country_name) + ", " + str(n_years) + " years \n -------------------------------------------------")

#------------------------------------------------------------------------------------------------

def modify_text_file( pathin ):
    '''This function is an example of how to modify a text file '''

    pathout = pathin.replace(".txt", "-aux.txt")

    for my_valuation_date in ["20200101"]:

        with open(pathin) as f:
            lines = f.readlines()

        file_out = open(pathout, "w+")
        for myline in lines:
            if not ("ValuationDate" in myline):
                file_out.write(myline)
            else:
                file_out.write(str("ValuationDate  " + my_valuation_date + "\n"))
        file_out.close()

    return

#------------------------------------------------------------------------------------------------

def create_file_fitting_parameters( input_params ):
    '''This function creates the (empty) dataframe where the parameters of the fitting to residuals will be stored.'''

    suffix = ''
    for distrib_type in input_params.list_distribution_types: suffix += "_" + str(distrib_type)
    infix = dict_ret[input_params.type_of_return]
    filepath = path.join(input_params.output_fit_dir, "Fitting_params_" + infix + str(input_params.list_product_labels[0]) + suffix + ".csv")

    if ( (not (input_params.calc_fitting) ) or (input_params.only_plots) ): return filepath

    list_cols = ["ret_type"]

    if ('norm' in input_params.list_distribution_types):          list_cols += ["normal_loc","normal_scale","normal_loss"]
    if ('nct'  in input_params.list_distribution_types):          list_cols += ["nct_loc", "nct_scale", "nct_skparam", "nct_dfparam", "nct_loss"]
    if ('levy_stable' in input_params.list_distribution_types):   list_cols += ["stable_loc", "stable_scale", "stable_beta_param", "stable_alpha_param", "stable_loss"]
    if ('genhyperbolic' in input_params.list_distribution_types): list_cols += ["ghyp_loc", "ghyp_scale", "ghyp_b_param","ghyp_a_param","ghyp_p_param", "ghyp_loss"]

    if (len(input_params.list_product_labels[1])==0): raise Exception("\n ERROR: The list of products is empty; check input.py.\n")

    df_fitting_params = pd.DataFrame( index=input_params.list_product_labels[1], columns=list_cols )
    df_fitting_params = df_fitting_params.rename_axis('Product_name')

    df_fitting_params.to_csv(filepath,index=True)

    del input_params; del df_fitting_params;  del suffix;  del list_cols

    return filepath

# ----------------------------------------------------------------------------------------------------------------------

def read_params_nct( directory, product_label ):
    '''This function reads the parameters of the fitting to an nct probability distribution which are stored in the "directory" (low starting with "product_label").'''

    from glob import glob

    nct_result_file_path = glob(directory+'/Fitting_params*nct*')
    if (nct_result_file_path==[]):
        nct_results = {'nct_loc': float('NaN')}
    else:
        df0 = pd.read_csv(nct_result_file_path[0],header=0)
        df0 = df0.set_index("Product_name")
        try:
            nct_results = df0.loc[product_label]
        except KeyError:
            nct_results = {'nct_loc': float('NaN')}

    del directory; del product_label; del nct_result_file_path; del df0

    return nct_results

# ----------------------------------------------------------------------------------------------------------------------

def read_params_from_file(product_label, pdf_type, type_of_return, output_fit_dir, only_plots=False, calc_fitting=False ):
    '''This function reads the parameters of the fitting to an nct probability distribution which are stored in the "directory" (low starting with "product_label").'''

    from glob import glob

    df0=None
    result_file_path = glob( output_fit_dir + '/Fitting_params*' + str( (pdf_type).replace("levy_","")) + '*')
    print("Now reading fitting parameters from", result_file_path[0])  # results_out

    if ( ( (not (calc_fitting)) or (only_plots))  and (not (path.exists(result_file_path[0]))) ):
        raise Exception("\nERROR: The path where the distribution parameters must be read does not exist. Please, calculate it.\n")

    if (result_file_path == []):
        results_out = {str(pdf_type) + '_loc': float('NaN')}
    else:

        df0 = pd.read_csv(result_file_path[0], header=0)
        df0 = df0.set_index("Product_name")

        try:
            results0 = df0.loc[product_label]
            if (pdf_type == "norm"):
                results_out = {"Product_name": product_label, "ret_type":type_of_return, 'distribution_type':pdf_type,
                               'loc_param': results0['normal_loc'], 'scale_param': results0['normal_scale']}
            elif (pdf_type == "nct"):
                results_out = {"Product_name": product_label, "ret_type":type_of_return, 'distribution_type':pdf_type,
                               'loc_param': results0['nct_loc'], 'scale_param': results0['nct_scale'],
                               'skewness_param': results0['nct_skparam'], 'df_param': results0['nct_dfparam']}
            elif (pdf_type == "genhyperbolic"):
                results_out = {"Product_name": product_label, "ret_type":type_of_return, 'distribution_type':pdf_type,
                               'loc_param': results0['ghyp_loc'], 'scale_param': results0['ghyp_scale'],
                               'a_param': results0['ghyp_a_param'], 'b_param': results0['ghyp_b_param'],
                               'p_param': results0['ghyp_p_param']}
            elif (pdf_type == "levy_stable"):
                results_out = {"Product_name": product_label, "ret_type":type_of_return, 'distribution_type':pdf_type,
                               'loc_param': results0['stable_loc'], 'scale_param': results0['stable_scale'],
                               'alpha_param': results0['stable_alpha_param'], 'beta_param': results0['stable_beta_param']}


        except KeyError:
            print("Could not read data.")
            results_out = {str(pdf_type) + '_loc': float('NaN')}

    if ((results_out["loc_param"]==None) or (results_out["scale_param"]==None)):
        raise Exception("\nERROR: The file which is expected to contain the fitted parameters is empty. Please, calculate it.\n")

    del product_label; del result_file_path; del df0

    return results_out

# ----------------------------------------------------------------------------------------------------------------------

def first_iteration(dataset_in, distrib_type, consider_skewness, loc_param0, sca_param0, skewness_param0, tail_param0, form_param0=None, consider_nonzero_p=False, verbose=0):
    '''This function provides the values to start the Barzilai-Borwein iterations.'''

    grad_form0 = None; form_param1 = None

    if (distrib_type == "nct"):
        from scipy.stats import nct
        from module_fitting_tstudent import calculate_gradient_params
        loss0 = - (np.sum(np.log(nct.pdf(dataset_in, loc=loc_param0, scale=sca_param0, nc=skewness_param0, df=tail_param0)))) / len(dataset_in)
        grad_loc0, grad_sca0, grad_sk0, grad_tail0 = calculate_gradient_params(dataset_in, consider_skewness,loc_param0, sca_param0,skewness_param0, tail_param0)
        factor_accept = 0.95; cauchy_scale = 0.001
    elif (distrib_type == "genhyperbolic"):
        from module_fitting_genhyperbolic import calculate_gradient_params
        from numpy.random import uniform
        from scipy.stats import genhyperbolic
        loss0 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param0, scale=sca_param0, b=skewness_param0, a=tail_param0,p=form_param0)))) / len(dataset_in)
        count = 0
        while ((isnan(loss0)) and (count<50) ):
            count+=1
            loc_param0 = 0
            sca_param0 = 2 * np.std(dataset_in[int(len(dataset_in) / 4): int(3 * len(dataset_in) / 4)])
            skewness_param0 = 0
            tail_param0 = uniform(1.01,1.99)#1.6
            form_param0 = 0
            loss0 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param0, scale=sca_param0, b=skewness_param0, a=tail_param0, p=form_param0)))) / len(dataset_in)
        if ((isnan(loss0)) and (count==50) ): raise Exception( "\nERROR: Could not find an appropriate starting guess for the generalized hyperbolic function. Please, try manually.\n")
        grad_loc0, grad_sca0, grad_sk0, grad_tail0, grad_form0 = calculate_gradient_params(dataset_in, consider_skewness, consider_nonzero_p, loc_param0, sca_param0, skewness_param0, tail_param0, form_param0 )
        factor_accept = 0.95; cauchy_scale = 0.001;
        for step in [ -10**(-7), 10**(-7), -10**(-8), 10**(-8),-3*10**(-8), 3*10**(-8), -3*10**(-7), 3*10**(-7), -10**(-6), 10**(-6), -10**(-5), 10**(-5) ]:
            loc_param1 = loc_param0 + step * grad_loc0
            sca_param1 = sca_param0 + step * grad_sca0
            skewness_param1 = skewness_param0 + step * grad_sk0
            tail_param1 = tail_param0 + step * grad_tail0
            form_param1 = form_param0 + step * grad_form0
            loss1 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param1, scale=sca_param1, beta=skewness_param1, alpha=tail_param1)))) / len(dataset_in)
            if not (isnan(loss1)):break
    elif (distrib_type == "levy_stable"):
        from scipy.stats import levy_stable
        from module_fitting_stable import calculate_gradient_params
        if (verbose > 0): print(" * Now doing the first iteration.")
        loss0 = - (np.sum(np.log(levy_stable.pdf(dataset_in, loc=loc_param0, scale=sca_param0, beta=skewness_param0,alpha=tail_param0)))) / len(dataset_in)
        if (verbose > 1): print("   Now calculating gradient of the first iteration.  ")
        grad_loc0, grad_sca0, grad_sk0, grad_tail0 = calculate_gradient_params(dataset_in, consider_skewness,  loc_param0, sca_param0, skewness_param0, tail_param0 )
        grad_loc0 = np.sign(grad_loc0) * min(500,abs(grad_loc0))
        grad_sca0 = np.sign(grad_sca0) * min(500,abs(grad_sca0))
        if (verbose>1): print("   The gradient is:",grad_loc0, grad_sca0, grad_sk0, grad_tail0)
        factor_accept = 0.95; cauchy_scale = 0.00005
    else:
        raise Exception("\nERROR: Unrecognized function"+distrib_type+"\n")
    skewness_param0 /= 2

    if (isnan(loss0)): # Unable to find an appropriate first step
        if (verbose>0): print("Unable to find an appropriate first step.")
        del dataset_in;del distrib_type;del consider_skewness;
        return loss0, loc_param0,sca_param0,skewness_param0,tail_param0,form_param0,None,None,None,None,None,9999,None,None,None,None,None

    if ( (distrib_type == "levy_stable") or ( (distrib_type == "genhyperbolic") and (isnan(loss1)) )):

        loss_opt = 999; step_opt = 9999
        for step in [ -10**(-8), 10**(-8),-3*10**(-8), 3*10**(-8), -10**(-7), 10**(-7), -3*10**(-7), 3*10**(-7), -10**(-6), 10**(-6), -10**(-5), 10**(-5) ]:
            loc_param1 = loc_param0 + step * grad_loc0
            sca_param1 = sca_param0 + step * grad_sca0
            skewness_param1 = skewness_param0 + step * grad_sk0
            tail_param1 = tail_param0 + step * grad_tail0
            loss1 = - (np.sum(np.log(levy_stable.pdf(dataset_in, loc=loc_param1, scale=sca_param1, beta=skewness_param1,alpha=tail_param1)))) / len(dataset_in)
            if (verbose > 0): print("   Step=", step,"; Loss0=", loss0,  "; Loss1=", loss1)
            if (loss1 < loss_opt):
                loss_opt = loss1
                step_opt = step
        if (loss_opt < factor_accept * loss0):
            loc_param1 = loc_param0 + step_opt * grad_loc0
            sca_param1 = sca_param0 + step_opt * grad_sca0
            skewness_param1 = skewness_param0 + step_opt * grad_sk0
            tail_param1 = tail_param0 + step_opt * grad_tail0
            loss1 = loss_opt

    if ((((distrib_type != "levy_stable") or (loss_opt >= factor_accept * loss0)  )) ):

        from scipy.stats import cauchy

        if (distrib_type != "genhyperbolic"): loss1=float('NaN')
        count = 0; max_count = 50
        while (count <= max_count) :
            if not (isnan(loss1)):
               if (loss1 < factor_accept * loss0):
                   break
            count += 1
            step = cauchy.rvs(size=1, loc=0, scale=cauchy_scale)[0]
            if (distrib_type == "levy_stable"):
                if ( (count > 12) and (count <= 20)):
                    step = ((-1) ** count) / (10 ** (4 + count / 2 - 6))
                elif ( count in [21,30,40,50,60] ) :
                    max_count = 70
                    cauchy_scale /= 10 # We do this because in some cases one finds huge gradients like: <<The gradient is: -691998447.1385866 -1.1509451688168508 -0.004771834087635612 -0.013121415046846715>>
                    factor_accept *= 0.8
            loc_param1 = loc_param0 + step * grad_loc0
            sca_param1 = sca_param0 + step * grad_sca0
            if (count<=100): skewness_param1 = skewness_param0 + step * grad_sk0
            if (count<=100): tail_param1 = tail_param0 + step * grad_tail0
            if (form_param0!=None): form_param1 = form_param0 + step * grad_form0

            if (distrib_type == "nct"):
                loss1 = - (np.sum(np.log(nct.pdf(dataset_in, loc=loc_param1, scale=sca_param1, nc=skewness_param1, df=tail_param1)))) / len(dataset_in)
            elif (distrib_type == "genhyperbolic"):
                loss1 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param1, scale=sca_param1, b=skewness_param1, a=tail_param1,p=form_param1)))) / len(dataset_in)
            elif (distrib_type == "levy_stable"):
                loss1 = - (np.sum(np.log(levy_stable.pdf(dataset_in, loc=loc_param1, scale=sca_param1, beta=skewness_param1,alpha=tail_param1)))) / len(dataset_in)
            if (verbose > 0): print(count, ") Loss0=", loss0, "; step=", step, "; Loss1=", loss1)

    if ( isnan(loss1)  ): # Unable to find an appropriate first step
        if (verbose>0): print("Loss1=",loss1,"; Loss0=",loss0,"; factor_accept=",factor_accept)
        del dataset_in;del distrib_type;del consider_skewness;
        return loss0, loc_param0,sca_param0,skewness_param0,tail_param0,form_param0,None,None,None,None,None,9999,None,None,None,None,None

    # If the loss of the new point is similar to that of the old point, yet higher, we swap the 0 and 1 points:
    if (loss1 > loss0):
        if (verbose>1): print("Swapping: loss0 ", loss0, "<--> Loss1", loss1)
        aux = loc_param1; loc_param1 = loc_param0; loc_param0 = aux
        aux = sca_param1; sca_param1 = sca_param0; sca_param0 = aux
        aux = skewness_param1; skewness_param1= skewness_param0; skewness_param0 = aux
        aux = tail_param1; tail_param1 = tail_param0; tail_param0 = aux
        aux = form_param1; form_param1 = form_param0; form_param0 = aux
        aux = loss1; loss1=loss0; loss0 = aux
        # Swapping points means that we have to recalculate the gradient at the point 0:
        if ((distrib_type == "nct") or (distrib_type == "levy_stable")):
            grad_loc0, grad_sca0, grad_sk0, grad_tail0 = calculate_gradient_params(dataset_in, consider_skewness,loc_param0, sca_param0,skewness_param0,tail_param0)
        elif (distrib_type == "genhyperbolic"):
            grad_loc0, grad_sca0, grad_sk0, grad_tail0, grad_form0 = calculate_gradient_params(dataset_in, consider_skewness, consider_nonzero_p, loc_param0, sca_param0,skewness_param0,tail_param0, form_param0)

    if (verbose > 0): print("     Parameters of the 1st iteration were found:",loc_param1,sca_param1,skewness_param1,tail_param1,";Loss1=",loss1)

    del dataset_in; del distrib_type; del consider_skewness; del cauchy_scale; del factor_accept; del consider_nonzero_p ; del verbose

    return loss0, loc_param0,sca_param0,skewness_param0,tail_param0,form_param0, grad_loc0, grad_sca0, grad_sk0, grad_tail0, grad_form0, loss1, loc_param1,sca_param1,skewness_param1,tail_param1,form_param1

# ----------------------------------------------------------------------------------------------------------------------

def first_iteration_single_param( sp, dataset_in, distrib_type, consider_skewness, lim_params, loc_param0, sca_param0, skewness_param0, tail_param0, form_param0=None):
    '''This function provides the values to start the Barzilai-Borwein iterations.'''

    from scipy.stats import cauchy

    if (distrib_type == "nct"):
        from scipy.stats import nct
        from module_fitting_tstudent import calculate_gradient_params, calculate_gradient_single_param
        loss0 = - (np.sum(np.log(nct.pdf(dataset_in, loc=loc_param0, scale=sca_param0, nc=skewness_param0, df=tail_param0)))) / len(dataset_in)
        my_grad = calculate_gradient_single_param(sp, dataset_in, consider_skewness, lim_params, loc_param0, sca_param0,skewness_param0, tail_param0)
    elif (distrib_type == "genhyperbolic"):
        from scipy.stats import genhyperbolic
        from module_fitting_genhyperbolic import calculate_gradient_params, calculate_gradient_single_param
        loss0 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param0, scale=sca_param0, b=skewness_param0, a=tail_param0,p=form_param0)))) / len(dataset_in)
        if (isnan(loss0)):
            loc_param0 = 0
            sca_param0 =  2*np.std(dataset_in[int(len(dataset_in) / 4): int(3 *len(dataset_in) / 4)])
            skewness_param0=0
            tail_param0=0
            form_param0=0
            loss0 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param0, scale=sca_param0, b=skewness_param0, a=tail_param0,p=form_param0)))) / len(dataset_in)
            if (isnan(loss0)):
                raise Exception("\nERROR: Could not find an appropriate starting guess for the generalized hyperbolic function. Please, try manually.\n")
        my_grad = calculate_gradient_single_param(sp, dataset_in, consider_skewness, lim_params, loc_param0, sca_param0, skewness_param0, tail_param0)
    elif (distrib_type == "levy_stable"):
        from scipy.stats import levy_stable
        from module_fitting_stable import calculate_gradient_params, calculate_gradient_single_param
        loss0 = - (np.sum(np.log(levy_stable.pdf(dataset_in, loc=loc_param0, scale=sca_param0, beta=skewness_param0,alpha=tail_param0)))) / len(dataset_in)
        my_grad = calculate_gradient_single_param(sp, dataset_in, consider_skewness, lim_params, loc_param0, sca_param0, skewness_param0, tail_param0)

    loss1=999; count = 0
    while (((loss1 > 0.995 * loss0) or (isnan(loss1))) and (count < 50)):

        count += 1
        step = cauchy.rvs(size=1, loc=0, scale=0.0005)[0]

        updp0 = {'a': tail_param0, 'b': skewness_param0}
        updp1 = {'a': tail_param0, 'b': skewness_param0}
        updp1[sp] += step * my_grad

        if (distrib_type == "nct"):
            loss1 = - (np.sum(np.log(nct.pdf(dataset_in, loc=loc_param0, scale=sca_param0, nc=updp1['b'], df=updp1['a'])))) / len(dataset_in)
        elif (distrib_type == "genhyperbolic"):
            loss1 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param0, scale=sca_param0, b=updp1['b'], a=updp1['a'],p=form_param0)))) / len(dataset_in)
        elif (distrib_type == "levy_stable"):
            loss1 = - (np.sum(np.log(levy_stable.pdf(dataset_in, loc=loc_param0, scale=sca_param0, beta=updp1['b'],alpha=updp1['a'])))) / len(dataset_in)

    if (count==50): # Unable to find an appropriate first step
        if not (distrib_type == "levy_stabe"):
            del dataset_in;del distrib_type;del consider_skewness; del loss0; del loss1
            return None,None,None
        else:
            for step in [-10 ** (-8), 10 ** (-8), -3 * 10 ** (-8), 3 * 10 ** (-8), -10 ** (-7), 10 ** (-7), -3 * 10 ** (-7), 3 * 10 ** (-7), -10 ** (-6), 10 ** (-6), -10 ** (-5), 10 ** (-5)]:
                updp0 = {'a': tail_param0, 'b': skewness_param0}
                updp1 = {'a': tail_param0, 'b': skewness_param0}
                updp1[sp] += step * my_grad
                loss1 = - (np.sum(np.log(levy_stable.pdf(dataset_in, loc=loc_param0, scale=sca_param0, beta=updp1['b'],alpha=updp1['a'])))) / len(dataset_in)
                if ((loss1 < 0.98 * loss0) and ( not isnan(loss1))): break
            if( isnan(loss1)):
                return None, None, None

    # If the loss of the new point is similar to that of the old point, yet higher, we swap the 0 and 1 points (including recalculation of the gradient at the point 0):
    if (loss1 > loss0):
        aux = updp1.copy(); updp1 = updp0.copy(); updp0 = aux.copy()
        if ((distrib_type == "nct") or (distrib_type == "levy_stable")):
            my_grad = calculate_gradient_single_param(sp, dataset_in, consider_skewness, lim_params, loc_param0, sca_param0, updp0['b'], updp0['a'] )
        elif (distrib_type == "genhyperbolic"):
            my_grad = calculate_gradient_single_param(sp, dataset_in, consider_skewness, lim_params, loc_param0, sca_param0, updp0['b'], updp0['a'])

    del dataset_in; del distrib_type; del consider_skewness; del loss0; del loss1

    return updp0, my_grad, updp1

# ----------------------------------------------------------------------------------------------------------------------

def find_truncation_limits_stocks( my_list_tickers=None ):
    '''This function calculates possible truncation limits for stocks using historical data downloaded from Yahoo Finance.'''

    from module_initialization import InputParams


    print(" *** NOW CALCULATING TRUNCATION LIMITS FOR STOCKS ***")

    if (my_list_tickers==None):
        my_list_tickers = [ ["Apple", ["AAPL"]], ["Microsoft", ["MSFT"]], ["ExxonMobil", ["XOM"]], ["SP500", ["^GSPC"]] ]

    for my_list in my_list_tickers:

        my_ticker = my_list[1][0]
        input_params = InputParams()  # Reading of the input parameters and initialization
        input_params.first_downloaded_data_date = "2018-06-23"  # Time range for downloading data
        input_params.last_downloaded_data_date = "2023-06-23"
        input_params.list_product_labels = my_list
        input_params.download_data = True
        input_params.download_ts()

        df0 = pd.read_csv( path.join(input_params.input_directory,"ret_"+my_ticker+".csv" ), header=0, usecols=["Date","rel_ret"] )
        my_max = df0["rel_ret"].max()
        my_min = df0["rel_ret"].min()
        print(my_ticker,": Min. rel. ret.=","{:.6f}".format(my_min),": Max. rel. ret.=","{:.6f}".format(my_max),"; data between",df0.loc[0,"Date"],"and",df0.loc[len(df0)-1,"Date"])

        # AAPL : Min. rel. ret.= -0.518692 : Max. rel. ret.= 0.332280 ; data between 1980-12-12 and 2023-06-16 <<<========
        #
        # MSFT : Min. rel. ret.= -0.301156 : Max. rel. ret.= 0.195652 ; data between 1986-03-13 and 2023-06-16
        # XOM :  Min. rel. ret.= -0.234285 : Max. rel. ret.= 0.179104 ; data between 1962-01-02 and 2023-06-16
        #SP500 : Min. rel. ret.= -0.254970 : Max. rel. ret.= 0.286374 ; data between 1927-12-30 and 2023-06-16
        # BP :   Min. rel. ret.= -0.191040 : Max. rel. ret.= 0.355872 ; data between 1962-01-02 and 2023-06-16 <<<========
        # Eni :  Min. rel. ret.= -0.218325 : Max. rel. ret.= 0.149273 ; data between 1995-11-28 and 2023-06-16
        # XOM :  Min. rel. ret.= -0.234286 : Max. rel. ret.= 0.179104 ; data between 1962-01-02 and 2023-06-16
        # SHEL : Min. rel. ret.= -0.171722 : Max. rel. ret.= 0.196795 ; data between 1994-10-31 and 2023-06-16
        # TTE :  Min. rel. ret.= -0.178208 : Max. rel. ret.= 0.152756 ; data between 1991-10-25 and 2023-06-23
        # COP :  Min. rel. ret.= -0.248401 : Max. rel. ret.= 0.252139 ; data between 1981-12-31 and 2023-06-23
        # CVX :  Min. rel. ret.= -0.221248 : Max. rel. ret.= 0.227407 ; data between 1962-01-02 and 2023-06-23
        # MPC :  Min. rel. ret.= -0.270089 : Max. rel. ret.= 0.206286 ; data between 2011-06-24 and 2023-06-23
        # MRO :  Min. rel. ret.= -0.468521 : Max. rel. ret.= 0.233568 ; data between 1962-01-02 and 2023-06-23 <<<========

    exit(0)

#------------------------------------------------------------------------------------------------

def find_truncation_limits_bonds( my_list_tickers=None ):
    '''This function calculates possible truncation limits for bonds using historical data.
    IMPORTANT: If you download the data from finanzen.net, then you may need to run a calculation in mode "fit" so that
    the dates are properly ordered (not conversely ordered).
    '''

    from module_initialization import InputParams

    print(" *** NOW CALCULATING TRUNCATION LIMITS FOR BONDS ***")

    if (my_list_tickers==None):
        my_list_tickers = [ ["First Republic Bank", ["US33616CAB63"]],["BASF",["XS1017833242"] ] ]

    for my_list in my_list_tickers:

        my_ticker = my_list[1][0]
        input_params = InputParams()  # Reading of the input parameters and initialization
        df0 = pd.read_csv( path.join(input_params.input_directory,my_ticker+".csv" ), header=0, usecols=["Date","Close"] )

        date0 = pd.to_datetime(df0.loc[0,"Date"])
        date1 = pd.to_datetime(df0.loc[1, "Date"])
        print(date0,date1);
        if (date0 > date1):
            factor = -1 # inverted dates => the max becomes the min, and the converse
        else:
            factor = 1
        df0['abs_ret'] = df0["Close"] - df0["Close"].shift(1)
        if (factor == 1 ):
            my_max = df0["abs_ret"].max()
            my_min = df0["abs_ret"].min()
        else:
            my_max = -df0["abs_ret"].min()
            my_min = -df0["abs_ret"].max()
        print(my_list[0],"-",my_ticker,": Min. abs. ret.=","{:.6f}".format(my_min),": Max. abs. ret.=","{:.6f}".format(my_max),"; data between",df0.loc[0,"Date"],"and",df0.loc[len(df0)-1,"Date"])

    # First Republic Bank - US33616CAB63 : Min. abs. ret.= -23.430000 : Max. abs. ret.= 8.560000 ; data between 20.06.2023 and 14.03.2023
    # BASF                - XS1017833242 : Min. abs. ret.= -3.460000 :  Max. abs. ret.= 2.900000 ; data between 10.06.2023 and 20.03.2014
    # https://www.finanzen.net/anleihen/historisch/a1v1s6-first-republic-bank-anleihe
    # Credit Suisse: XS0989394589 https://www.boerse-frankfurt.de/bond/xs0989394589-ubs-group-ag-7-5: Se ve caida max en un dia de 39.78 - 78.34 = -38.56: Eurobonos: Credit Suisse Group AG, 7.5% perp., USD (1) XS0989394589 https://cbonds.es/bonds/67043/
    # https://www.finanzen.net/anleihen/a19tlb-charles-schwab-anleihe

    exit(0)



#------------------------------------------------------------------------------------------------

