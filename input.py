
# ====================================================================================================
#                           INPUT PARAMETERS
# ====================================================================================================

# This is the file which contains the parameters defined by the user

from numpy import  array, log

#-------------------------------------------------------------------------------------------------------------------

calculation_mode = "ES_HS-vs-N"    #"fit" # Must be either None, "fit", "ES_HS-vs-N", "ES_distrib-vs-N" or "ES_distrib-vs-N_last_dates"
                                    # "fit": Fitting of the input data to given probability density functions.
                                    # "ES_HS-vs-N": Generation of synthetic data from the fitted distributions, and plotting
                                    #               of the NUMERICAL ES (calculated from discrete data, NOT directly from
                                    #               distribution parameters) vs the number of datapoints of the sample (N).
                                    # "ES_distrib-vs-N": Like, "ES_HS-vs-N" but in this case the program also calculates the ES
                                    #               from integrating the probability density function.
                                    # "ES_distrib-vs-N_last_dates" is like "ES_distrib-vs-N" (it compares continuous and discrete)
                                    #               yet the data are NOT synthetic data, but the observed returns of the last N dates FOR STOCKS
                                    #               (throughout the whole time series).

list_distribution_types = ["genhyperbolic"] # "norm", "nct", "genhyperbolic", "levy_stable"

array_N_values         = array([10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800])  # array( [10, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500 ]  )
array_N_values_distrib = array([20, 30, 50, 75, 100, 125, 150, 200, 250, 300 ])

# Number of times that the ES is calculated for every sample size
n_random_trials_calcES         = 500000 # Recommended value for "ES_HS-vs-N": 500000 (en el main paper es half a million)
n_random_trials_calcES_distrib = 1000   #  Recommnended value: 2000; minimum value:40


# Number of times that each set of starting parameters in the fitting of the random variable is tried. Each trial corresponds to a different first iteration of the gradient descent algorithm.
n_random_trials_fit = 8   # Recommended value for "fit" calculations: 8; recommended value for "ES_distrib-vs-N" calculations: lower (e.g. 3).
n_random_trials_fit_levy = 6 # Suggested value for levy_stable: Min 3 !!

# Max number of iterations in each search for optimal parameters (using the gradient descent algorithm, with a given starting value for the distribution parameters).
max_n_iter = 50  # Recommended value for "fit" calculations: 30; recommended value for "ES_distrib-vs-N" calculations: lower.
max_n_iter_levy = 16  # Suggested value for levy_stable: Depends on the size of the sample (smaller sample, higher number of necessary iterations); at least 10 !!!

# To be used if (calculation_mode == "ES_distrib-vs-N"):
n_random_trials_fit_distrib = 2
n_random_trials_fit_levy_distrib = 1
max_n_iter_distrib = 20
max_n_iter_levy_distrib = 10


#directory_input_data = r"/Users/pablogarciarisueno/Desktop/Finanzas/Fin_my_own_papers/paper_ES_executions/Input_data/Corporate_bonds/"
#directory_output_data = r"/Users/pablogarciarisueno/Desktop/Finanzas/Fin_my_own_papers/paper_ES_executions/Output_data/Corporate_bonds/"
directory_input_data = r"/Users/pgr/Desktop/Ciencia/IT/Python_my_own_programs/PaperES/Input_data/Corporate_bonds/"
directory_output_data = r"/Users/pgr/Desktop/Ciencia/IT/Python_my_own_programs/PaperES/Output_data/Corporate_bonds/"

# IMPORTANT: Run the function find_truncation_limits with a ticker of a product analogous to your product to determine wise values of the truncation limits below. In the code of the functions you can see example limits.
top_limit_bonds     =  30  # These are the truncation limits for the synthetic data; every generated random number which exceeds them will be discarded.
bottom_limit_bonds  = -30  # IMPORTANT: If the time series is a price of a bond which corresponds to a notional about 100 currency units, this should be of the order of tens; if, conversely, the notional is about one, then this should be of the order of one tenth (0.1).
top_limit_stocks    =  log(1.40) # log(1.33) for apple!
bottom_limit_stocks =  log(0.60)  # log(0.48) for apple!

product_type = "stocks" # Either "stocks" or "bonds". If "stocks", then the logarithmic return of the corrected (due to reinvested dividends) price is used, regardless of other variables.
list_product_labels = [ "AAPL",["AAPL"] ]  # ["SHELL.AS",["SHELL.AS"]] # [ "BASF_bond", ["XS1017833242"]] #   ["CharlesSchwab_bond",["US808513AL92"] ] # [ "BASF_bond", ["XS1017833242"]] # ["CharlesSchwab_bond",["US808513AL92"] ] #   [ "AAPL",["AAPL"] ]  # ["SHELL.AS",["SHELL.AS"]] #  [ "AAPL",["AAPL"] ]  #  ["SHELL.AS",["SHELL.AS"]]# [ "AAPL",["AAPL"] ]  # ["CharlesSchwab_bond",["US808513AU91"] ] # ["Eni",["E"]] # ["BP",["BP"]] # ["ExxonMobil",["XOM"]] # "Shell",["SHELL.AS"]# ["First Republic Bank", ["US33616CAB63"]] # [ "apple",["AAPL"] ] # [ "BASF_bond", ["XS1017833242"]] ["CharlesSchwab_bond",["US808513AU91"] ]
li_filenames = list_product_labels[1]
kind_of_price = "Close"

# Set to more than 0 (i.e. 1 or 2) to print partial results from screen
verbose = 1

make_plots = True
only_plots = True

ratio_aux = 20.0 # This is the number which is used to determine the size of an auxiliary array which is used to refill the array of synthetic data after removing outliers.

# BLOCK FOR DOWNLOADING DATA OF STOCKS (from Yahoo Finance; using the yfinance Python module)
download_data = False                      # If True, the code will download the data of the Yahoo Finance tickers written in list_product_labels
first_downloaded_data_date = "2018-06-22"  # Time range for downloading data
last_downloaded_data_date  = "2023-06-24"