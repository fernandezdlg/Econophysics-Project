#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:06:32 2019

@author: juanfernandez
"""

import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

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
            
            
                
def main():
    data = openFileAsPanda()
    
    #data['Close'].plot()
    print(data.shape[0])
    
    volume = np.zeros(data.shape[0])
    
    #data['datetime'] = data['date'].map(lambda x: datetime.datetime.strptime(x, ))
    data['Assets'] = data['Close'] * data['Volume']
    data['Return'] = data['Close'].diff()
    data.plot('Unix Timestamp', 'Return')
    
    
    
    """
    for i in range(data.shape[0]):
        volume[i] = data['Close'][i] * data['Volume'][i]

    plt.plot(volume)
    plt.show()
    """
    
main()   
    
    
    
    
    
    
    