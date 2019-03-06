import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from functools import reduce

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
    with open("gemini_BTCUSD_2015_1min.csv") as csv_file:
        data = pd.read_csv(csv_file)
        print("Prices in USD imported.")
    
    with open("USACPI.csv") as csv_file:
        inflation = pd.read_csv(csv_file)
        print("Inflation imported.")
        
    return data, inflation
    
# returns the factors of a number n
def factors(n):    
    return set(reduce(list.__add__, # this part adds each factor to a list
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))     

def factors2(n,a): # returns a factors from number n
    multiplier = n**(1/a)
    factors = np.zeros(a)
    for i in range(a):
        factors[i] = np.int(np.floor(multiplier**(i+1)))
    return factors
            
                
def main():
    # The prices of cryptocurrencies in USD are imported
    data, inflation = openFileAsPanda()
    
    #data['Close'].plot()
    print(data.shape[0])
    print(inflation.shape[0])
    
    volume = np.zeros(data.shape[0])
    
    #data['datetime'] = data['date'].map(lambda x: datetime.datetime.strptime(x, ))
    data['Formatted Date'] = [dt.datetime.strptime(date,'%d/%m/%Y %H:%M') for date in data['Date']]
    data['Assets'] = data['Close'] * data['Volume']
    data['Return'] = data['Close'].diff()
    
    length_max = len(data.index)
    fa = np.array(factors(length_max)) # Hope that length is not a prime number
    print(fa)
#==============================================================================
#     Instead of getting factors of the number, we could plan to get a certain
#     number of factors for the Hurst plot and the round off to integers as
#     showed below:
#==============================================================================
    fa2 = factors2(length_max,10)
    print(fa2)
    print('sample start')
    for i in range(10):
        print(np.zeros(np.int(fa2[i])))
        
    print('sample ends')
    
    plt.close()
    #data.plot('Formatted Date', 'Assets')
    
    ####### Idea so far:
    # Find factors of the total length of list - 121580
    # Use this as block length to calculate Hurst exponent 
    
    rav = []
    for n in np.int(fa2):
        r = []
        for k in range(int(length_max/n)):
            S = np.std(x[k*n:(k+1)*n])
            R = max(X[k*n:(k+1)*n]) - min(X[k*n:(k+1)*n])
            r.append(R/S)
        rav.append(np.mean(r))
        
    lnN = np.log(np.int(fa2))
    lnrav = np.log(rav)
    
    plnNlnrav = np.polyfit(lnN,lnrav,1)
    lnravfit = np.polyval(plnNlnrav,lnN)


    
    """
    for i in range(data.shape[0]):
        volume[i] = data['Close'][i] * data['Volume'][i]

    plt.plot(volume)
    plt.show()
    """
    
main()   
    
    
    
    
    
    
    