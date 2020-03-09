import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.sequence import SequenceSet
from quantiprot.utils.sequence import subset, columns
from quantiprot.utils.feature import Feature, FeatureSet

# Conversions-related imports:
from quantiprot.utils.mapping import simplify
from quantiprot.metrics.aaindex import get_aa2charge, get_aa2hydropathy
from quantiprot.metrics.aaindex import get_aaindex_file
from quantiprot.metrics.basic import identity

import numpy as np
import pandas as pd


def getLabelIndex(fromFile):
	splitArray = fromFile.split(' ', 1)

	# Remove last element from split
	splitArray.pop(0)
	y = splitArray[0].split(' ')
	# y = y[:-1]
	y = sorted([int(i) for i in y[:-1]])

	return y


# Load the 'xxxxx.fasta' sequence set
alphasyn_seq = load_fasta_file("sequence.fasta")
alphasyn_seq1 = load_fasta_file("sequence.fasta")

# Get array of lengths
fastaLength = []

for seq in alphasyn_seq1: fastaLength.append(len(seq.data))

print(len(fastaLength))

count = 0
for leng in fastaLength: count += leng

print(count)

# Set of features
fs = FeatureSet("Basic Features")

# Add feature names to set of features

# ZIMMERMAN HYDROPHOBICITY
fs.add(get_aaindex_file("ZIMJ680101"))

# AVERAGE FLEXIBILITY INDICES
fs.add(get_aaindex_file("BHAR880101"))

# HOPP-WOOD HYDROPHILICITY
fs.add(get_aaindex_file("HOPT810101"))

# GRANTHAM POLARITY
fs.add(get_aaindex_file("GRAR740102"))

# CONFORMATIONAL PARAMETER OF BETA STRUCTURE
fs.add(get_aaindex_file("BEGF750102"))
# print(fs)

conv_seq = fs(alphasyn_seq)


# # OPEN FILE AND READING FILE
f = open("glenn_labels.txt","r")
lines = f.readlines()

# Convert to matrix and print

# ZIMMERMAN HYDROPHOBICITY
mat1 = np.matrix(columns(conv_seq, feature="ZIMJ680101", transpose=True))

# AVERAGE FLEXIBILITY INDICES
mat2 = np.matrix(columns(conv_seq, feature="BHAR880101", transpose=True))

# HOPP-WOODS HYDROPHILICITY
mat3 = np.matrix(columns(conv_seq, feature="HOPT810101", transpose=True))

# GRANTHAM POLARITY
mat4 = np.matrix(columns(conv_seq, feature="GRAR740102", transpose=True))

# CONFORMATIONAL PARAMETER OF BETA STRUCTURE
mat5 = np.matrix(columns(conv_seq, feature="BEGF750102", transpose=True))

# START LOOP HERE

firstIteration = True
allMatrix = np.array([])
for k in range(len(fastaLength)):
	newmat = np.concatenate((mat1[k],mat2[k],mat3[k],mat4[k], mat5[k]))
	newmat1 = (np.transpose(newmat[:,:fastaLength[k]]))

	# CHANGE EVERY LOOP

	# y - SORTED INDICES

	#ADD HERE
	indice = getLabelIndex(lines[k]) 

	# RESET LABELS
	label = []

	j = 0 
	for i in range(fastaLength[k]):
		if(i == indice[j]-1): 
			label.append(1)
			if(j != len(indice) - 1):
				j += 1
		else: label.append(0)

	# Convert to numpy array
	labels = np.array(label)

	# Convert column vector
	col_vec = labels.reshape(-1, 1)

	# print(newmat1.shape, col_vec.shape)
	# print(col_vec)

	# Combine two matrices
	finalMatrix = np.append(newmat1, col_vec, axis=1)

	if firstIteration: 
		allMatrix = finalMatrix
		firstIteration = False
	else: allMatrix = np.concatenate((allMatrix, finalMatrix))

	print(k)
	# print(finalMatrix)

	# # END LOOP


# SAVE TO CSV
	# df = pd.DataFrame(finalMatrix)
	# fileName = "newmat" + str(k+1) + ".csv"
	# df.to_csv(fileName, index=False, header = ["ZIMJ680101", "BHAR880101","HOPT810101","GRAR740102","BEGF750102", "Labels"])

print(allMatrix)

df = pd.DataFrame(allMatrix)
fileName = "newmat.csv"
df.to_csv(fileName, index=False, header = ["ZIMJ680101", "BHAR880101","HOPT810101","GRAR740102","BEGF750102", "Labels"])
