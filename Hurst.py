import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from functools import reduce
import tradingeconomics as te # requires pip install tradingeconomics

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
    
        print("File opened!")
        data = pd.read_csv(csv_file)
    
    return data
    
# returns the factors of a number n
def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))        
            
                
def main():
    # The prices of cryptocurrencies in USD are imported
    data = openFileAsPanda()
    # The inflation of USD is imported
    te.login('n1800703j@e.ntu.edu.sg:PH4410-ECONOPHYSICS')
    print(te.getHistoricalData(country='United States', indicators='Inflation Rate', initDate='2015-01-01'))
#    plt.plot(inflation)
    
    #data['Close'].plot()
    print(data.shape[0])
    
    
    volume = np.zeros(data.shape[0])
    
    #data['datetime'] = data['date'].map(lambda x: datetime.datetime.strptime(x, ))
    data['Formatted Date'] = [dt.datetime.strptime(date,'%d/%m/%Y %H:%M') for date in data['Date']]
    data['Assets'] = data['Close'] * data['Volume']
    data['Return'] = data['Close'].diff()
    
    length_max = len(data.index)
    fa = array(factors(length_max))
    print(fa)
    
    plt.close()
    #data.plot('Formatted Date', 'Assets')
    
    ####### Idea so far:
    # Find factors of the total length of list - 121580
    # Use this as block length to calculate Hurst exponent 
    
    
    
    
    """
    for i in range(data.shape[0]):
        volume[i] = data['Close'][i] * data['Volume'][i]

    plt.plot(volume)
    plt.show()
    """
    
main()   
    
    
    
    
    
    
    