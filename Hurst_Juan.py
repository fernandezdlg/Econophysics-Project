import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from functools import reduce
import matplotlib

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
        
def openFileAsPanda(year):
    with open("gemini_BTCUSD_"+str(year)+"_1min.csv") as csv_file:
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

def hurstFitPlot(lnN, lnrav, lnravfit, year):
    plt.close('all')
    matplotlib.rcParams.update({'font.size': 22})
    plt.plot(lnN, lnravfit, 'r')
    plt.plot(lnN, lnrav, 'o')
    plt.xlabel('ln(facs)')
    plt.ylabel('ln(R/S)')
    plt.title('Bitcoin '+str(year))
    plt.show()

    
            
###########################################                
#### MAIN CODE FOR EXECUTING FUNCTIONS HERE
###########################################    

# The prices of cryptocurrencies in USD are imported
# Inflation is also imported
year = 2018
data, inflat = openFileAsPanda(year)

# change date so it can be plotted
inflat['Formatted Date'] = [dt.datetime.strptime(date, '%Y-%m') for date in inflat['TIME']]



if year == 2015:
    data['Formatted Date'] = [dt.datetime.strptime(date,'%d/%m/%Y %H:%M') for date in data['Date']]
elif year > 2015:
    data['Formatted Date'] = [dt.datetime.strptime(date,'%Y-%m-%d %H:%M:%S') for date in data['Date']]


#print(inflat['Value'])
#print(data['Formatted Date'][10])


"""
# code below adds inflation to 'data' DataFrame but it is extremely slow
data['Inflation'] = 0.0
for dataIndex, dataRow in data.iterrows():
    dataMonth = dataRow['Formatted Date'].month
    
    print(dataIndex)
    
    
    
    for inflatIndex, inflatRow in inflat.iterrows():
        inflatYear, inflatMonth = inflatRow['Formatted Date'].year, inflatRow['Formatted Date'].month
        
        print(inflatRow['Value'])
        
        if inflatYear == year and inflatMonth == dataMonth:
            inflatValue = inflatRow['Value']
            data['Inflation'][dataIndex] = inflatValue
"""        


# This code aims to add inflation in a more efficient way
# Define more natural inflation
nat_inflat = 1+inflat.Value/100
# Change it to a cummulative one
for i in range(1,inflat.Value.size):
    nat_inflat[i] = nat_inflat[i]*nat_inflat[i-1]


# Extract Month and Year from inflation
inflat['Formatted Date'] = pd.to_datetime(inflat['Formatted Date'])
inflat['year'], inflat['month'] = inflat['Formatted Date'].dt.year, inflat['Formatted Date'].dt.month



month_loc = where(inflat.year == year)
for i in range(1,12):
    linear_inflat = linspace(nat_inflat[month_loc[i]-1],nat_inflat[month_loc[i]],)
linear_inflat = linspace(nat_inflat[month_loc[0]],nat_inflat)


# add a 'Returns' column
data['Returns'] = data['Close'].diff()


# Get array with sizes of sections in time series 
length_max = len(data.index)
#facs = factors(length_max)
facs = factors_a(length_max,10)    


# calculate returns based on data stored 
# generally the first element in returns will be a NaN 
# replace NaN with 0.
rets = np.array(data['Returns'])
where_are_NaNs = np.isnan(rets)
rets[where_are_NaNs] = 0.


"""
rets_dates = np.array(data['Formatted Date'])
rets_timestamp = np.array(data['Unix Timestamp'])

print(rets_timestamp)
for timestamp in rets_timestamp:
    dt.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m'))
"""


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
        
        # occasionally S will be stored as a very small number ~e-13 instead of zero
        # occurs due the rets[k*fac:(k+1)*fac] elements being equal
        # this leads to errors where R/S will be 'inf' or extremely large
        # This if statement counters this by using a new equation to calculate standard deviation
        if R/S == float('inf') or R/S > 1000000:
            S = abs(rets[k*fac])/(fac**0.5)
                        
        r.append(R/S)    
        
    r = np.array(r)
    
    # give zero, instead of NaN, if S standard deviation is zero
    where_are_NaNs = np.isnan(r)
    r[where_are_NaNs] = 0.
      
    rav.append(np.mean(r))
    

# log both lists
lnN = np.log(facs)
lnrav = np.log(rav)

# fit lnN as x and lnrav as y as a linear fit
# in accordance with equation given in 'Statstical Properties of Financial Time Series'
plnNlnrav = np.polyfit(lnN,lnrav,1)
lnravfit = np.polyval(plnNlnrav,lnN)  

# print values of coefficients of linear fit
print(plnNlnrav)

# plot lnrav and lnravfit 
hurstFitPlot(lnN, lnrav, lnravfit, year) 