import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from functools import reduce

plt.close('all')

def openFile():
    with open("gemini_BTCUSD_2015_1min.csv") as csv_file:
    
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
        
def openFileAsPanda():
    with open("gemini_BTCUSD_2018_1min.csv") as csv_file:
        data = pd.read_csv(csv_file)
        print("Prices in USD imported.")
    
    with open("USACPI.csv") as csv_file:
        inflation = pd.read_csv(csv_file)
        print("Inflation imported.")
        
    return data, inflation
    
# returns the factors of a number n
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
            
###########################################                
#### MAIN CODE FOR EXECUTING FUNCTIONS HERE
###########################################    

# The prices of cryptocurrencies in USD are imported
# Inflation is also imported
data, inflation = openFileAsPanda()

# change date so it can be plotted
# date change for 2015 file
#data['Formatted Date'] = [dt.datetime.strptime(date,'%d/%m/%Y %H:%M') for date in data['Date']]
# date change for 2018 file
#data['Formatted Date'] = [dt.datetime.strptime(date,'%Y-%m-%d %H:%M:%S') for date in data['Date']]

# add a 'Returns' column
#data['Assets Traded'] = data['Close'] * data['Volume']
data['Returns'] = data['Close'].diff()

length_max = len(data.index)
#==============================================================================
#     Plan to get a certain number of factors for the Hurst plot
#     round off to integers as
#     showed below:
#==============================================================================
# Get array with sizes of sections in time series 
#facs = factors(length_max)
facs = factors_a(length_max,10)    




#### Not sure what this commented out code means?
#    print('sample start')
#    for i in range(10):
#        print(np.zeros(np.int(fa2[i])))
#        
#    print('sample ends')
#    
#    plt.figure()
#    data.plot('Formatted Date', 'Assets')
#####



# calculate returns based on data stored 
# generally the first element in returns will be a NaN 
# replace NaN with 0.
rets = np.array(data['Returns'])
where_are_NaNs = np.isnan(rets)
rets[where_are_NaNs] = 0.

# cumulative sum of returns
rets_cumsum = np.cumsum(rets)

# list storing average of R/S for section sizes given (given as factors)
rav = []

# calculation of <R(fac)/S(fac)> for different section sizes fac
for fac in facs:
    r = []
    
    S_3 = []
    
    
    for k in range(np.int(length_max/fac)): 
        S = np.std(rets[k*fac:(k+1)*fac])
        R = np.max(rets_cumsum[k*fac:(k+1)*fac]) - np.min(rets_cumsum[k*fac:(k+1)*fac])
        
        if R/S == float('inf') or R/S > 100000:
            S = abs(rets[k*fac]) / (fac**0.5)
                        
            
            print('#####')
            print(R)
            print(S)
            print(R/S)
            print(fac)
            print(k)
            print(rets[k*fac:(k+1)*fac])
            print(abs(rets[k*fac]/ (fac**0.5)))
            print(np.max(rets_cumsum[k*fac:(k+1)*fac]))
            print(np.min(rets_cumsum[k*fac:(k+1)*fac]))
            print('#####')
        
        r.append(R/S)
    
    ### Find out what is causing massive spike in rav for fac == 3
    # firstly find minimum value in S_3 (standard deviations from fac==3) list, which is not zero
    # idea is to find out if an extremely small S value is causing R/S to be large.
    # find out what values are causing it
        
    
        
        
    r = np.array(r)
    
    # give zero, instead of NaN, if S standard deviation is zero
    where_are_NaNs = np.isnan(r)
    r[where_are_NaNs] = 0.
    
    
    rav.append(np.mean(r))
    

 
# log both lists
lnN = np.log(facs)
lnrav = np.log(rav)


plt.figure()
plt.plot(lnN,lnrav)

# fit lnN as x and lnrav as y as a linear fit
# in accordance with equation given in 'Statstical Properties of Financial Time Series'
plnNlnrav = np.polyfit(lnN,lnrav,1)
lnravfit = np.polyval(plnNlnrav,lnN)  

# print values of coefficients of linear fit
print(plnNlnrav)
    