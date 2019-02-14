#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:06:32 2019

@author: juanfernandez
"""

import csv

<<<<<<< HEAD
with open("gemini_BTCUSD_2015_1min.csv") as csv_file:
    
    print("File opened!")
    
    csv_reader = csv.reader(csv_file, delimiter = ",")
    line_count = 0
    
    
  
    
    for row in csv_reader:
        if line_count == 1:
            #print(f'Column names are {", ".join(row)}')
            line_count += 1
            
        elif line_count > 1:
            #print(f'\t{row[0]} is timestamp, {row[1]} is date and {row[2]} is symbol.')
            line_count += 1
            
        elif line_count == 0:
            # skips over line_count = 0
            # stops parsing over first line in .csv file
            line_count += 1
    print("Done!")
=======
file = open('gemini_BTCUSD_2015_1min.csv')
>>>>>>> 3788a6faec3f7df440e9ae825fcb903406f314bc
