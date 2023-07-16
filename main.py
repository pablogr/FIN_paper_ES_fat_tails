'''
This is the main script for the calculations on a possible paper to prove how the
historical simulation systematically underestimates the ES.

Author: Pablo Risueno, 2021-2023
'''

from module_initialization import InputParams
from module_fitting import fit_to_distributions
from module_ES_vs_N import ES_calculations, ES_calculations_last_dates

# WE ANALYSE THE STOCKS OF APPLE (AAPL) AND SHELL (EUROPEAN, SHELL.AS)
# AND TWO BONDS OF BASF (XS1017833242) AND CHARLES SCHWAB (US808513AU91).
# Thresholds:
# For the bonds, +/-30 (abs ret). This is an orientation; recently a bond of First Republic Bank fell 23 (abs ret) in one day.
# For the stocks: Since AAPL fell 52% in a day (Sept. 2000), we take the extrema from AAPL as threshods: log ret corresponding to rel rets of -0.518692, +0.332280
# For Shell, we take the limits among the extreme relative returns of a series of oil companies: Max: BP, 0.355872; min: MRO: -0.468521.


from module_plots import plot_comparison
my_dir = r'/Users/pgr/Desktop/Finanzas/Papers_Finanzas_IM/Paper1_ES_errors/Datos_paper_ES/2023-07-15/'
plot_comparison( my_dir,  "ES_HS_log_ret_AAPL-nct_summary.csv","ES_HS_log_ret_AAPL-ghyp_summary.csv", "non-centered t-student", "generalized hyperbolic","AAPL")
exit(0)

if __name__ == '__main__':

    # Reading of the input parameters and initialization
    input_params = InputParams( )

    # Fitting of data to probability density functions
    fit_to_distributions(input_params)

    # Calculations of the expected shortfall
    ES_calculations( input_params, "ES_HS-vs-N" )      # Without fitting of new distributions to datasets
    ES_calculations( input_params, "ES_distrib-vs-N")  # With fitting of new distributions to datasets
    ES_calculations_last_dates(input_params)                # From last dates


'''



from module_generic_functions import find_truncation_limits_stocks

find_truncation_limits_stocks(  [  [ "Shell",["SHELL.AS"] ] ] )
exit(0)

'''

''' 
from module_plots import plot_ES_vs_N, plot_bare_histograms

#plot_ES_vs_N( input_params.output_directory, "ES_HS_abs_ret_corporate_bonds_summary.csv")
#plot_bare_histograms( input_params, "2023-06-12_ES_HS_abs_ret_corporate_bonds.csv")
plot_bare_histograms( input_params, "2023-06-15_Avg_HS_abs_ret_corporate_bonds.csv","Avg")
exit(0)
'''



''' 
***** 2021 VERSION *****

print_message_1()
directory_input_data = r"/Users/pgr/Desktop/Ciencia/IT/Python_my_own_programs/PaperES/Input_data/Government_bonds/"
directory_output_data = r"/Users/pgr/Desktop/Ciencia/IT/Python_my_own_programs/PaperES/Output_data/"
is_bond = True

for country_name in ["Italy"]: #xxx [ "Portugal", "Greece" ,  "Spain", "Italy", "Netherlands", "France",  "Germany", "United Kingdom", "Poland" ]:
    for n_years in [5]: #xxx[  10, 5, 1 ]:

       filename_ts_input = f"{country_name} { str(n_years) }-Year Bond Yield Historical Data"
       if not path.exists(f"{directory_input_data}{filename_ts_input}.csv"): continue
       print_message_2(country_name,  str(n_years) )

       mydataset = FittedTimeSeries( directory_input_data, filename_ts_input, directory_output_data, "Price", "nct", is_bond, n_years )
       mydataset.fit_to_distribution()
       mydataset.plot_fitting()

       mydataset = FittedTimeSeries( directory_input_data, filename_ts_input, directory_output_data, "Price", "genhyperbolic", is_bond, n_years , False )
       mydataset.fit_to_distribution()
       mydataset.plot_fitting()

       mydataset = FittedTimeSeries( directory_input_data, filename_ts_input, directory_output_data, "Price", "genhyperbolic", is_bond, n_years , True )
       mydataset.fit_to_distribution()
       mydataset.plot_fitting()

'''









'''
p, a, b, loc, scale = 1, 1, 0, 0, 1
x = np.linspace(-10, 10, 100)
print(genhyperbolic.pdf(x, p, a, b, loc, scale) - genhyperbolic.pdf(x, loc=loc, scale=scale, b=b, a=a, p=p ))

plt.figure(0)


#plt.plot(x,  genhyperbolic.pdf(x, p, a, b, loc, scale), lw=4, color='r',label = 'GH(p=1, a=1, b=0, loc=0, scale=1)')
#plt.plot(x,  genhyperbolic.pdf(x, p, a, b, loc, scale), lw=1.5, color='b',label = 'GH(p=1, a=1, b=0, loc=0, scale=1)')

plt.title("Generalized Hyperbolic | -10 < p < 10")
plt.plot(x,  genhyperbolic.pdf(x, p, a, b, loc, scale),
        label = 'GH(p=1, a=1, b=0, loc=0, scale=1)')
plt.plot(x,  genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'red', alpha = 0.5, label='GH(p>1, a=1, b=0, loc=0, scale=1)')
[plt.plot(x, genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'red', alpha = 0.1) for p in np.linspace(1, 10, 10)]
plt.plot(x, genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'blue', alpha = 0.5, label='GH(p<1, a=1, b=0, loc=0, scale=1)')
[plt.plot(x, genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'blue', alpha = 0.1) for p in np.linspace(-10, 1, 10)]
# plt.ylim(1e-15, 1e2)
# plt.yscale('log')
plt.legend(bbox_to_anchor=(1.1, 1))
plt.subplots_adjust(right=0.5)

# plot GH for different values of a
plt.figure(1)
plt.title("Generalized Hyperbolic | 0 < a < 10")
plt.plot(x, genhyperbolic.pdf(x, p, a, b, loc, scale),
        label = 'GH(p=1, a=1, b=0, loc=0, scale=1)')
plt.plot(x, genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'blue', alpha = 0.5, label='GH(p=1, a>1, b=0, loc=0, scale=1)')
[plt.plot(x, genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'blue', alpha = 0.1) for a in np.linspace(1, 10, 10)]
plt.plot(x, genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'red', alpha = 0.5, label='GH(p=1, 0<a<1, b=0, loc=0, scale=1)')
[plt.plot(x, genhyperbolic.pdf(x, p, a, b, loc, scale),
        color = 'red', alpha = 0.1) for a in np.linspace(0, 1, 10)]
#plt.ylim(1e-15, 1e2)
#plt.yscale('log')
plt.legend(bbox_to_anchor=(1.1, 1))
plt.subplots_adjust(right=0.5)


plt.show()

exit()
'''


# np.random.seed(123)
mysize = 2000
# myvar = nct.rvs( size=mysize, loc=0, scale=1, nc=0, df=2.25 )
# myvar = genhyperbolic.rvs(size=mysize, loc=0, scale=1, b=0, a=0.4, p=0)
# print("skew",skew(myvar))

''' Generation of synthetic input file (if you use this, make sure that you do L35 of generic functions: df0["abs_ret"] = df0[field_to_read] 
myloc = -1
myscale = 0.5
np_ret = nct.rvs(size=mysize, loc=myloc, scale=myscale, nc=0, df=2.5)
np_ret = genhyperbolic.rvs(size=mysize, loc=myloc, scale=myscale, b=0, a=1, p=0)
# plot_histogram0(np_ret, "prueba", "nct", myloc, myscale)
df0 = pd.DataFrame(np_ret)
df0 = df0.reset_index()
df0.columns = ["Date","Price"]
directory_input_data = r"/Users/pgr/Desktop/Ciencia/IT/Python_my_own_programs/PaperES/Input_data/Government_bonds/"
df0.to_csv(f"{directory_input_data}pru.csv",index=False )
'''

#---------------------------------------------------------------------------------------------------------------------

'''
print_message_1()
directory_input_data = r"/Users/pgr/Desktop/Ciencia/IT/Python_my_own_programs/PaperES/Input_data/Government_bonds/"
directory_output_data = r"/Users/pgr/Desktop/Ciencia/IT/Python_my_own_programs/PaperES/Output_data/"

for country_name in [ "Greece", "Portugal", "Netherlands", "France", "Spain", "Italy", "Germany", "United Kingdom", "Poland"]:
    for n_years in ["1", "5", "10"]:

       filename_ts_input = f"{country_name} {n_years}-Year Bond Yield Historical Data"
       if not path.exists(f"{directory_input_data}{filename_ts_input}.csv"): continue
       print_message_2(country_name, n_years)

       mydataset = FittedTimeSeries( directory_input_data, filename_ts_input, directory_output_data, "Price", "nct" )
       mydataset.fit_to_distribution()
       mydataset.plot_fitting()

       mydataset = FittedTimeSeries( directory_input_data, filename_ts_input, directory_output_data, "Price", "genhyperbolic", False )
       mydataset.fit_to_distribution()
       mydataset.plot_fitting()

       mydataset = FittedTimeSeries( directory_input_data, filename_ts_input, directory_output_data, "Price", "genhyperbolic", True )
       mydataset.fit_to_distribution()
       mydataset.plot_fitting()

exit()
'''

'''


exit()


myvar1 = np.sort( nct.rvs( size=5000000, loc=0, scale=0.05, nc=0, df=2.5 ) )
print("Numerical",calc_ES_discrete(myvar1) )

for Nin in [ 50000000]:
  print( "Analyt", calc_ES_continuous(Nin) )


exit()
 
for mysize in [10,20,30,40,50,60,70,80,100,120,140,160,180,200,300,350,400,450,500,600,700,800,900,1000,1200,1400]:

    mysize *=100
    myvar = nct.rvs(size=mysize, loc=0, scale=1, nc=0, df=2.5) #+ genhyperbolic.rvs(size=mysize, loc=0, scale=1, b=0, a=0.33, p=0)
    myvar = np.sort(myvar)
    ES = calc_ES_discrete(myvar)
    print( mysize, ";", ES )
    #fit_to_nct_global_minimum(myvar)
    #fit_to_genhyperbolic_global_minimum(myvar)
    #print("=====================\n")


exit()


'''

