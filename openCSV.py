import os
import sys

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.abspath('..'))

# mycsv = pd.read_csv("./Feature Selection/feature-selection-dataset.csv")

# print(mycsv)

# mycsv.drop(mycsv.columns[16], axis=1, inplace=True)

# print(mycsv.columns)

# mycsv.to_csv('fs-dataset.csv', index = False)

data = pd.read_csv("./Feature Selection/fs-dataset.csv")

print(data)

y = data.pop('Labels')

print(y)

print("Before OverSampling, counts of label '1': {}".format(sum(y == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y == 0))) 

sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_sample(data, y.ravel())

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
print("After OverSampling, counts of label '0': {} \n".format(sum(y_train_res == 0))) 

print(X_train_res)
print(y_train_res)

X_train_res['Labels'] = y_train_res

print(X_train_res)

X_train_res.to_csv('bl-fs-dataset.csv', index = False)

data = pd.read_csv("bl-fs-dataset.csv")

print(data)