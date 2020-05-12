import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('..'))

mycsv = pd.read_csv("./Feature Selection/feature-selection-dataset.csv")

print(mycsv)

mycsv.drop(mycsv.columns[16], axis=1, inplace=True)

print(mycsv.columns)

mycsv.to_csv('fs-dataset.csv', index = False)

newcsv = pd.read_csv("fs-dataset.csv")

print(newcsv)