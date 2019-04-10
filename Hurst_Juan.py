import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np        # Ensure latest update of numpy is installed
import datetime as dt
from functools import reduce
import matplotlib
import matplotlib.dates as mdates
import arch #pip install arch

#==============================================================================
# Fancy plotting
#==============================================================================
import matplotlib.pylab as pylab
# To plot with Serif font
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
# Plotting parameters
params = {'legend.fontsize':'small',
          'figure.figsize': (12, 6),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize':'medium',
          'ytick.labelsize':'medium'}
pylab.rcParams.update(params)

plt.close('all')


def openFile(year):
    with open("gemini_BTCUSD_"+str(year)+"_1min.csv") as csv_file:
    
        print("File opened!")
    
        csv_reader = csv.reader(csv_file, delimiter = ",")
        line_count = 0
    
    
        for row in csv_reader:
            if line_count == 1:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
                
            elif line_count == 0:
                # skips over line_count = 0
                # stops parsing over first line in .csv file
                line_count += 1
            
            elif line_count > 1:
                #print(f'\t{row[0]} is timestamp, {row[1]} is date and {row[2]} is symbol.')
                print(row)
                line_count += 1
            
        print("Done!")
           
#==============================================================================
# Import all data for Bitcoin    
#==============================================================================

def openFileAsPanda():
    dfs = []
    for k in range(4):
        with open("gemini_BTCUSD_"+str(2018-k)+"_1min.csv") as csv_file:
            dfs.append(pd.read_csv(csv_file))
            
            # To create consistency betweeen date formats
            if k == 3:
                dfs[k]['Formatted_Date'] = [dt.datetime.strptime(date,'%d/%m/%Y %H:%M') for date in dfs[k]['Date']]
            else:
                dfs[k]['Formatted_Date'] = [dt.datetime.strptime(date,'%Y-%m-%d %H:%M:%S') for date in dfs[k]['Date']]
            
            print(str(2018-k) + " BTC prices in USD imported.")
    
    # Concatenate    
    data = pd.concat(dfs, ignore_index=True)
    
    with open("USACPI.csv") as csv_file:
        inflation = pd.read_csv(csv_file)
        print("Inflation imported.")
        
    return data, inflation
    
#==============================================================================
# Returns the factors of a number n
#==============================================================================
def factors(n):    
    # this part adds each factor to a list
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))     

# returns a factors from number n
def factors_a(n,a): 
    multiplier = n**(1/a)
    factors = np.zeros(a)
    for i in range(a):
        factors[i] = multiplier**(i+1)

    # Convert to array of integers
    return factors.astype(int)

#==============================================================================
# Fit to Hurst exponent
#==============================================================================
def fitHurst(facs,length,rets):
    # cumulative sum of returns
    rets_cumsum = np.cumsum(rets)
    # list for storing average of R/S for section sizes given (given as factors)
    rav = []
    # calculation of <R(fac)/S(fac)> for different section sizes fac
    for fac in facs:
        r = [] # stores partial R/S        
        for k in range(np.int(length/fac)): 
            S = np.std(rets[k*fac:(k+1)*fac])
            R = np.max(rets_cumsum[k*fac:(k+1)*fac]) - np.min(rets_cumsum[k*fac:(k+1)*fac])
            
            # occasionally S will be stored as a very small number ~e-13 instead of zero
            # occurs due the rets[k*fac:(k+1)*fac] elements being equal
            # this leads to errors where R/S will be 'inf' or extremely large
            # This if statement counters this by using a new equation to calculate standard deviation
            if R/S == float('inf') or R/S > 1000000: # filter nans
                S = abs(rets[k*fac])/(fac**0.5)  # correction formula
                            
            r.append(R/S)    
            
        r = np.array(r)
        
        # give zero, instead of NaN, commonly happens if S standard deviation is zero
        where_are_NaNs = np.isnan(r)
        r[where_are_NaNs] = 0.
        print(r.shape)
        if r.shape[0] == 0:
            r = 2*rav[fac-1]-rav[fac-2] # In case only nans are detected
        
        rav.append(np.mean(r))
        
    
    # Fitting
    # log both lists
    lnN = np.log(facs)
    lnrav = np.log(rav)
    
    # fit lnN as x and lnrav as y as a linear fit
    # in accordance with equation given in 'Statstical Properties of Financial Time Series'
    plnNlnrav = np.polyfit(lnN,lnrav,1)
    lnravfit = np.polyval(plnNlnrav,lnN)  
    
    # print values of coefficients of linear fit
    print(plnNlnrav)
    return lnN,lnrav,lnravfit,plnNlnrav[0]


#==============================================================================
# Plots plot of fitting
#==============================================================================
def hurstFitPlot(lnN, lnrav, lnravfit):
    plt.figure()
    matplotlib.rcParams.update({'font.size': 22})
    plt.plot(lnN, lnravfit, 'r')
    plt.plot(lnN, lnrav, 'o')
    rav_error = np.std(abs(lnrav-lnravfit))#/sqrt(len(lnrav))
    plt.errorbar(lnN,lnrav,yerr=rav_error, xerr=None, fmt='none')
    plt.xlabel(r'$\ln\, N$')
    plt.ylabel(r'$\ln \left\langle R/S \right\rangle$')
#    plt.title('Bitcoin ') # Leave title parameters outside the plot
    plt.show()

         
###########################################                
#### MAIN CODE FOR EXECUTING FUNCTIONS HERE
###########################################    

# The prices of cryptocurrencies in USD are imported
# Inflation is also imported
data, inflat = openFileAsPanda()

# change date so it can be plotted
inflat['Formatted_Date'] = [dt.datetime.strptime(date, '%Y-%m') for date in inflat['TIME']]

# Get Range for Months and Years of interest
YYMM = np.array([[data['Formatted_Date'].dt.year[0],data['Formatted_Date'].dt.year[data.shape[0]-1]],[data['Formatted_Date'].dt.month[0],data['Formatted_Date'].dt.month[data.shape[0]-1]]])

# This code aims to add inflation in a more efficient way
# Define more natural inflation
nat_inflat = (1 + inflat.Value/100)**(1/12)
# Change it to a 'cummulative' one
for i in range(1,inflat.Value.size):
    nat_inflat[i] = nat_inflat[i]*nat_inflat[i-1]


# Extract Month and Year from inflation
inflat['Formatted_Date'] = pd.to_datetime(inflat['Formatted_Date'])
inflat['year'], inflat['month'] = inflat['Formatted_Date'].dt.year, inflat['Formatted_Date'].dt.month

year_loc_min = np.where(inflat.year == YYMM[0,1])[0]
year_loc_max = np.where(inflat.year == YYMM[0,0])[0]

month_loc_min = year_loc_min[np.where(inflat.month[year_loc_min] == YYMM[1,1])[0]]
month_loc_max = year_loc_max[np.where(inflat.month[year_loc_max] == YYMM[1,0])[0]]

if (month_loc_min.size == 0) or (month_loc_min[0] == 0):
    print('Not enough information about inflation available')
    month_loc_min = np.array([1])
if (month_loc_max.size == 0):
    month_loc_max = np.array([len(inflat.month)-1])

month_loc = np.arange(month_loc_min[0],month_loc_max[0]+1)
    
# Instead of for loop and month sizes finder, it assumes each month is more or less the same size:
linear_inflat = np.interp(np.linspace(0,1,data.shape[0]),np.linspace(0,1,len(nat_inflat[np.append(month_loc[0]-1,month_loc)])),nat_inflat[np.append(month_loc[0]-1,month_loc)])

linear_inflat = linear_inflat[::-1] # Reverse order to match prices ordering

data['Close'] = data['Close']*linear_inflat   # De-inflation of prices

# Already managed by .dropna()
data['pct_change'] = data['Close'].pct_change().dropna()
data['pct_change'][0] = 0. #NaN value

data['Returns'] = 100 * data['pct_change'] # Using this definition of returns seem to be better due to independence of scale

# Store returns in array
rets = np.array(data['Returns'])

"""
rets_dates = np.array(data['Formatted_Date'])
rets_timestamp = np.array(data['Unix Timestamp'])

print(rets_timestamp)
for timestamp in rets_timestamp:
    dt.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m'))
"""

# Get array with sizes of sections in time series 
length_max = len(data.index)
#facs = factors(length_max)
facs = factors_a(length_max,10)    

lnN,lnrav,lnravfit,plnNlnrav = fitHurst(facs,length_max,rets)

#==============================================================================
# # plot lnrav and lnravfit for all BTC prices, demonstrate the need for a parametrization of the Hurst exponent
#==============================================================================
hurstFitPlot(lnN, lnrav, lnravfit) 
plt.title('Hurst exponent estimation for all BTC prices')
plt.tight_layout()

#plt.figure()
#data[['Formatted_Date','Close']].plot()
#plt.title('All prices for BTC')
#plt.ylabel('Inflation-weighted USD')
#plt.xlabel('Time')


#==============================================================================
# Find parametrization for Hurst exponent
#==============================================================================
width = 60*24*30*6  # width of each interval to find Hurst exponent
shared = 0.9  # how much is shared between two neighboring intervals
jump = np.int(width*(1-shared))
N = np.int(length_max/jump)-1
Npoints = 10
mlnN = np.zeros([N,Npoints])
mlnrav = np.zeros([N,Npoints])
mlnravfit = np.zeros([N,Npoints])
mplnNlnrav = np.zeros(N)

for n in range(N):
    facs = factors_a(width,2*Npoints)[Npoints:-2]
#    print(rets[jump*n:jump*n+width])
#    a,b,c = fitHurst(facs,width,rets[jump*n:jump*n+width])
    mlnN[n,:],mlnrav[n,:],mlnravfit[n,:],mplnNlnrav[n] = fitHurst(facs, width, rets[jump*n:jump*n+width])
     
    print(str(1+n) + '/' + str(N) +' fittings done')
        
    hurstFitPlot(mlnN[n,:], mlnrav[n,:], mlnravfit[n,:]) 
#    plt.title(n)
#    plt.tight_layout()
#    plt.figure()


data['Hurst1']=np.interp(np.linspace(0,1,data.shape[0]),np.linspace(0,1,len(mplnNlnrav)),mplnNlnrav)
plt.figure()
plt.plot(data.Formatted_Date,data.Hurst1)


#==============================================================================
# Second returns definition
#==============================================================================

data['Returns'] = data['Close'].diff()
rets = np.array(data['Returns'])
where_are_NaNs = np.isnan(rets) # replace NaN with 0.
rets[where_are_NaNs] = 0.

for n in range(N):
    facs = factors_a(width,2*Npoints)[Npoints:-2]
#    print(rets[jump*n:jump*n+width])
#    a,b,c = fitHurst(facs,width,rets[jump*n:jump*n+width])
    mlnN[n,:],mlnrav[n,:],mlnravfit[n,:],mplnNlnrav[n] = fitHurst(facs, width, rets[jump*n:jump*n+width])
     
    print(str(1+n) + '/' + str(N) +' fittings done')
        
    hurstFitPlot(mlnN[n,:], mlnrav[n,:], mlnravfit[n,:]) 
#    plt.title(n)
#    plt.tight_layout()
#    plt.figure()


data['Hurst2']=np.interp(np.linspace(0,1,data.shape[0]),np.linspace(0,1,len(mplnNlnrav)),mplnNlnrav)
plt.figure()
plt.plot(data.Formatted_Date,data.Hurst2)

#plt.title('Hurst exponent fitting for the Returns of Bitcoin')


    


'''
#==============================================================================
# GARCH FITTING
#==============================================================================

data['stdev21'] = data['pct_change'].rolling(21*24*60).std() #rolling window stdev
data['hvol21'] = data['stdev21']*((360*24*60)**0.5) # Annualized volatility
data['variance'] = data['hvol21']**2
data = data.dropna() # Remove rows with blank cells.
#data.head()

#data['Returns'][79360] = 0. # Remove wierd value
am=np.zeros(N).tolist()
res=np.zeros(N).tolist()
params=np.zeros(N).tolist()

for n in range(N):
    am[n] = arch.arch_model(rets[jump*n:jump*n+width] * 100)
#    res = am.fit(update_freq=5)
    res[n] = am[n].fit()
    params[n] = res[n].params

#==============================================================================
# I think that the fitting is not working because the model parameters vary with time, shorter time intervals are needed
#==============================================================================
'''


