'''
This module contains the functions to read the input files.
'''

import input
from os import path, makedirs
from numpy import array, log
import pandas as pd
#import yfinance as yf


#path.append(getcwd())

#-----------------------------------------------------------------------------------------------------------------------

class InputParams:
    '''This is a data class which contains the values of the parameters determined by the user for the calculations.'''

    def __init__(self, price_from_yield=False, time_to_maturity=None, consider_p=True,consider_skewness=True ):

        '''
        :param directory_input_data: (string) Directory to read the input time series
        :param directory_output_data: (string) Directory where the output files and plots will be stored
        :param list_distribution_types: (list of strings) List of functions to be fitted to.
        :param price_from_yield: (Boolean, optional) Set to True if the time series to be analysed corresponds to yields (not to prices)
        :param time_to_maturity: (int, optional) Time to maturity of government bonds
        :param type_of_return: (string, optional) either "absolute", "relative" or "logarithmic".
        :param consider_p: (Boolean, optional) Set to True if fitting to genhyperbolic considering p parameter must be carried out
        :param consider_skewness: (Boolean, optional) Set to True if fitting to nct, levy_stable or genhyperbolic considering skewness must be carried out
        '''

        # PARAMETERS FROM input.py
        try: self.calculation_mode = input.calculation_mode
        except: self.calculation_mode = None
        assert self.calculation_mode in [ None, "fit", "ES_HS-vs-N", "ES_distrib-vs-N", "ES_distrib-vs-N_last_dates"]
        if (self.calculation_mode in [ None, "fit"]): self.calc_fitting=True
        else: self.calc_fitting = False
        if (self.calculation_mode in [ None, "ES_HS-vs-N"]): self.calc_ES_HS = True
        else: self.calc_ES_HS =  False
        if (self.calculation_mode in [ None, "ES_distrib-vs-N"]): self.calc_ES_distrib = True
        else: self.calc_ES_distrib =  False

        try: self.array_N_values = input.array_N_values  # Values of N that will be considered in the calculations.
        except AttributeError: self.array_N_values = array( [10,30,100 ]  )

        self.n_random_trials_calcES = input.n_random_trials_calcES
        self.input_directory  = input.directory_input_data
        self.ts_directory     = self.input_directory  # + "/Time_series"
        self.output_directory = input.directory_output_data
        self.directory_output_plots = self.output_directory+"/Plots"
        self.output_fit_dir = self.output_directory + "/Results"
        for filepath in [self.output_directory, self.output_fit_dir, self.directory_output_plots, path.join(self.directory_output_plots, "Histograms")]:
            if not (path.exists(filepath)): makedirs(filepath)
        self.list_distribution_types = input.list_distribution_types
        self.price_from_yield = price_from_yield
        self.product_type = input.product_type.lower()
        assert self.product_type in ["stocks","bonds"]
        if (self.product_type == "stocks"):
            self.type_of_return = "logarithmic"
            # self.calc_ES_distrib = True # Since the stocks are very liquid, we only analyse them in mode "ES_distrib-vs-N_last_dates"
        else:# "bonds"
            self.type_of_return = "absolute"
        self.time_to_maturity = time_to_maturity
        self.consider_p = consider_p
        self.consider_skewness = consider_skewness
        self.list_product_labels = input.list_product_labels
        self.first_downloaded_data_date = input.first_downloaded_data_date
        self.last_downloaded_data_date = input.last_downloaded_data_date
        self.download_data = input.download_data

        if (self.product_type == "stocks"):
            self.top_limit    = input.top_limit_stocks
            self.bottom_limit = input.bottom_limit_stocks
        else: # if (self.product_type == "bonds")
            self.top_limit    = input.top_limit_bonds
            self.bottom_limit = input.bottom_limit_bonds

        try: self.ratio_aux = input.ratio_aux
        except: self.ratio_aux = 15

        try:
            self.verbose = input.verbose  # If set to 1 or 2 intermediate results of the calculations are printed to screen.
        except AttributeError:
            self.verbose = 0

        try:
            self.make_plots = input.make_plots  # Set to False if plots are not to be made in the execution.
        except AttributeError:
            self.make_plots = True

        try:
            self.only_plots = input.only_plots  # Set to True to just make the plots, without any calculation.
        except AttributeError:
            self.only_plots = False

        self.make_plots = input.make_plots
        self.only_plots = input.only_plots
        # Number of times that each set of starting parameters in the fitting of the random variable is tried. Each trial corresponds to a different first iteration of the gradient descent algorithm.
        try:
            self.n_random_trials_fit = input.n_random_trials_fit
        except AttributeError:
            self.n_random_trials_fit = 200
        try:
            self.n_random_trials_fit_levy = input.n_random_trials_fit_levy
        except AttributeError:
            self.n_random_trials_fit_levy = 2

        # Max number of iterations in each search for optimal parameters (using the gradient descent algorithm, with a given starting value for the distribution parameters).
        try:
            self.max_n_iter = input.max_n_iter
        except AttributeError:
            self.max_n_iter = 100  # If the optimal solution was not reached in 100 iterations, probably it will not be improved in further 100 iterations.
        try:
            self.max_n_iter_levy = input.max_n_iter_levy
        except AttributeError:
            self.max_n_iter_levy = 1

        # OTHER PARAMETERS
        #self.print_intial_message()  Descomentar e incluir mensaje
        self.fit_data = True

        # Kind of price to be used
        self.field_to_read = input.kind_of_price # Name of the colum of the file named filename to be analised (e.g. "Price", "Open" or "Close").
        assert self.field_to_read in [ None, "Open", "High", "Low", "Close", "Price", "Yield" ]
        if (self.field_to_read == None): self.kind_of_price = "Close"

        if (self.download_data):
            self.download_ts()

        print("\n====================================================================")
        print("                  NOW ANALYSING " + (str(self.list_product_labels[0]).upper()))
        print("====================================================================")

        return


    def download_ts(self):
        '''This function downloads time-series from Yahoo Finance and stores them to files.'''

        for product_label in self.list_product_labels[1]:
            print("\n\n===================== Now downloading", product_label, "=====================\n")

            # Download
            dwl_data = yf.download(product_label, start=self.first_downloaded_data_date, end=self.last_downloaded_data_date, actions=True)

            # Correction to 2 decimal places
            # for colname in ["Open","High","Low","Close","Dividends"]:
            #    if (colname in list(dwl_data)):
            #        dwl_data[colname] = round(100 * dwl_data[colname]) / 100

            # Calculation of returns and storage
            if not (dwl_data.empty):
                if (dwl_data.index.names == ['Datetime']): dwl_data.index.names = ['Date']
                dwl_data.to_csv(self.input_directory + "/ts_" + product_label + ".csv", index=True)
                print(dwl_data)
            else:
                print("\nSEVERE WARNING: Could not download the data of", product_label)
                dwl_data = pd.read_csv(self.input_directory + "/ts_" + product_label + ".csv", header=0)

            self.calculate_returns( product_label, dwl_data )


    def calculate_returns(self, product_label, dwl_data):
        '''This function receives a dataframe with prices, calculates their returns and stores them to file.
        It includes a column of 'corrected price' which includes dividends assuming that they are reinvested.
        This is, for each ex-dividend date the relative return ('rel_ret_with_dividends') is calculated using
        the non-corrected prices and the dividend. Then the corrected price is calculated as the previous
        corrected price times this relative return.
        '''

        field_to_read = self.field_to_read  # e.g. "Close", "Open", ...

        if (("Stock Splits" in list(dwl_data)) and (abs(dwl_data["Stock Splits"].sum()) > 0.00000001)):
            my_index = dwl_data["Stock Splits"].max()
            df_aux = pd.DataFrame(dwl_data["Stock Splits"])
            df_aux = df_aux.reset_index()
            df_aux = df_aux.set_index("Stock Splits");
            df_aux.columns = ["Date"]
            print("\n * Split for", product_label, "(", my_index, ") on;", df_aux.loc[my_index, "Date"], "\n")
            del df_aux; del my_index

        if (not ("Dividends" in list(dwl_data))):
            print("\nSEVERE WARNING: No dividends found for", product_label, "\n")
            dwl_data["Dividends"] = 0.0

        my_col = "price_" + field_to_read
        my_col_c = my_col + "_corrected"
        my_col_c_log = "log_" + my_col + "_corrected"
        dwl_data = pd.DataFrame(dwl_data[[field_to_read, "Dividends"]])
        dwl_data.columns = [my_col, "Dividends"]

        dwl_data['abs_ret'] = dwl_data[my_col] - dwl_data[my_col].shift(1)
        dwl_data['rel_ret'] = (dwl_data[my_col] - dwl_data[my_col].shift(1)) / dwl_data[my_col].shift(1)
        dwl_data['rel_ret_with_dividends'] = (dwl_data[my_col] + dwl_data["Dividends"] - dwl_data[my_col].shift(1)) / \
                                             dwl_data[my_col].shift(1)

        dwl_data[my_col_c] = None
        dwl_data[my_col_c_log] = None
        dwl_data.loc[dwl_data.index[0], my_col_c] = dwl_data.loc[dwl_data.index[0], my_col]

        for i in range(1, len(dwl_data)):
            dwl_data.loc[dwl_data.index[i], my_col_c] = (dwl_data.loc[dwl_data.index[i - 1], my_col_c]) * (
                        1 + dwl_data.loc[dwl_data.index[i], 'rel_ret_with_dividends'])
            dwl_data.loc[dwl_data.index[i], my_col_c_log] = log(dwl_data.loc[dwl_data.index[i], my_col_c])

        # Calculation of returns (abs, rel, log) of the column of interest
        dwl_data[my_col_c + '_abs_ret'] = dwl_data[my_col_c] - dwl_data[my_col_c].shift(1)
        dwl_data[my_col_c + '_rel_ret'] = (dwl_data[my_col_c] - dwl_data[my_col_c].shift(1)) / dwl_data[my_col_c].shift(1)
        dwl_data[my_col_c + '_log_ret'] = dwl_data[my_col_c] / dwl_data[my_col_c].shift(1)
        for ind in dwl_data.index:
            if (isinstance(dwl_data.loc[ind, my_col_c + '_log_ret'], float) or isinstance( dwl_data.loc[ind, my_col_c + '_log_ret'], int)):
                dwl_data.loc[ind, my_col_c + '_log_ret'] = log(dwl_data.loc[ind, my_col_c + '_log_ret'])
            else:
                dwl_data.loc[ind, my_col_c + '_log_ret'] = None

        dwl_data.to_csv(self.input_directory + "/ret_" + product_label + ".csv", index=True)

        del product_label; del dwl_data; del field_to_read; del my_col

        return

# -----------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------

