import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('..'))

data = pd.read_csv("./Training Model/training1-20.csv")
print(data)

data.to_csv('combined-dataset.csv', index = False, mode='w', header = None)



data1 = pd.read_csv("./Training Model/training21-40.csv")
print(data1)

data1.to_csv('combined-dataset.csv', index = False, mode='a', header = None)

#

data2 = pd.read_csv("./Training Model/training41-60.csv")
print(data2)

data2.to_csv('combined-dataset.csv', index = False, mode='a', header = None)

#

data3 = pd.read_csv("./Training Model/training61-80.csv")
print(data3)

data3.to_csv('combined-dataset.csv', index = False, mode = 'a', header = None)

#

data4 = pd.read_csv("./Training Model/training81-100.csv")
print(data4)

data4.to_csv('combined-dataset.csv', index = False, mode = 'a', header = None)

#

data5 = pd.read_csv("./Training Model/training101-120.csv")
print(data5)

data5.to_csv('combined-dataset.csv', index = False, mode = 'a', header = None)

#

data6 = pd.read_csv("./Training Model/training121-140.csv")
print(data1)

data6.to_csv('combined-dataset.csv', index = False, mode = 'a', header = None)

#

data7 = pd.read_csv("./Training Model/training141-156.csv")
print(data7)

data7.to_csv('combined-dataset.csv', index = False, mode = 'a', header = None)

#

data8 = pd.read_csv("./Training Model/training157-180.csv")
print(data8)

data8.to_csv('combined-dataset.csv', index = False, mode = 'a', header = None)


mycsv = pd.read_csv("combined-dataset.csv")
print(mycsv)

headerName = ["ZIMJ680101","BHAR880101","HOPT810101","GRAR740102","BEGF750102","JOND750101","KARP850101","PRAM900101","KUHL950101","SWER830101","RADA880108","JANJ780101","ZIMJ680103","BEGF750103","ARGP820101","RADA880106","JANJ780101","CHOP780101","KARP850102","KARP850103","VINM940102","VINM940103","VINM940104", "Labels"]
mycsv.to_csv('feature-selection-dataset.csv', index = False, header = headerName)

mycsv = pd.read_csv("feature-selection-dataset.csv")
labels = mycsv['Labels']

print("Before OverSampling, counts of label '1': {}".format(sum(labels == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(labels == 0))) 
print("Testing commit")

