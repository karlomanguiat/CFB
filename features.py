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
	y = sorted([int(i) for i in y[:-1]])

	return y


# Load the 'xxxxx.fasta' sequence set
alphasyn_seq = load_fasta_file("./Sequences/train1-20.fasta")
alphasyn_seq1 = load_fasta_file("./Sequences/train1-20.fasta")

# Get array of lengths
fastaLength, count = [], 0

for seq in alphasyn_seq1: 
	fastaLength.append(len(seq.data))

for leng in fastaLength: 
	count += leng


# Set of features
fs = FeatureSet("Basic Features")

# Add feature names to set of features

# Hydrophobicity (Zimmerman et al., 1968)
fs.add(get_aaindex_file("ZIMJ680101"))

# Average flexibility indices (Bhaskaran-Ponnuswamy, 1988)
fs.add(get_aaindex_file("BHAR880101"))

# Hydrophilicity value (Hopp-Woods, 1981)
fs.add(get_aaindex_file("HOPT810101"))

# Polarity (Grantham, 1974)
fs.add(get_aaindex_file("GRAR740102"))

# Conformational parameter of beta-structure (Beghin-Dirkx, 1975)
fs.add(get_aaindex_file("BEGF750102"))

# Hydrophobicity (Jones, 1975)
fs.add(get_aaindex_file("JOND750101"))

# Flexibility parameter for no rigid neighbors (Karplus-Schulz, 1985)
fs.add(get_aaindex_file("KARP850101"))

# Hydrophobicity (Prabhakaran, 1990)
fs.add(get_aaindex_file("PRAM900101"))

# Hydrophilicity scale (Kuhn et al., 1995)
fs.add(get_aaindex_file("KUHL950101"))

# Optimal matching hydrophobicity (Sweet-Eisenberg, 1983)
fs.add(get_aaindex_file("SWER830101"))

conv_seq = fs(alphasyn_seq)


# # OPEN FILE AND READING FILE
f = open("./Sequences/training_labels2","r")
lines = f.readlines()
print("Length of lines: %i" % len(lines))
# Convert to matrix and print

# Hydrophobicity (Zimmerman et al., 1968)
mat1 = np.matrix(columns(conv_seq, feature="ZIMJ680101", transpose=True))

# Average flexibility indices (Bhaskaran-Ponnuswamy, 1988)
mat2 = np.matrix(columns(conv_seq, feature="BHAR880101", transpose=True))

# Hydrophilicity value (Hopp-Woods, 1981)
mat3 = np.matrix(columns(conv_seq, feature="HOPT810101", transpose=True))

# Polarity (Grantham, 1974)
mat4 = np.matrix(columns(conv_seq, feature="GRAR740102", transpose=True))

# Conformational parameter of beta-structure (Beghin-Dirkx, 1975)
mat5 = np.matrix(columns(conv_seq, feature="BEGF750102", transpose=True))

# Hydrophobicity (Jones, 1975)
mat6 = np.matrix(columns(conv_seq, feature="JOND750101", transpose=True))

# Flexibility parameter for no rigid neighbors (Karplus-Schulz, 1985)
mat7 = np.matrix(columns(conv_seq, feature="KARP850101", transpose=True))

# Hydrophobicity (Prabhakaran, 1990)
mat8 = np.matrix(columns(conv_seq, feature="PRAM900101", transpose=True))

# Hydrophilicity scale (Kuhn et al., 1995)
mat9 = np.matrix(columns(conv_seq, feature="KUHL950101", transpose=True))

# Optimal matching hydrophobicity (Sweet-Eisenberg, 1983)
mat10 = np.matrix(columns(conv_seq, feature="SWER830101", transpose=True))

print(mat1, len(mat1), "1")
print(mat2, len(mat2), "2")
print(mat3, len(mat3), "3")
print(mat4, len(mat4), "4")
print(mat5, len(mat5), "5")

print(mat6, len(mat6), "6")
print(mat7, len(mat7), "7")
print(mat8, len(mat8), "8")
print(mat9, len(mat9), "9")
print(mat10, len(mat10), "10")


# START LOOP HERE

firstIteration = True
allMatrix = np.array([])
for k in range(1):
	newmat = np.concatenate((mat1[k],mat2[k],mat3[k],mat4[k], mat5[k], mat6[k],mat7[k],mat8[k],mat9[k], mat10[k]))
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
	print(finalMatrix)

	# # END LOOP


# # SAVE TO CSV
# 	# df = pd.DataFrame(finalMatrix)
# 	# fileName = "newmat" + str(k+1) + ".csv"
# 	# df.to_csv(fileName, index=False, header = ["ZIMJ680101", "BHAR880101","HOPT810101","GRAR740102","BEGF750102", "Labels"])

# print(allMatrix)

# df = pd.DataFrame(allMatrix)
# fileName = "newmat.csv"
# df.to_csv(fileName, index=False, header = ["ZIMJ680101", "BHAR880101","HOPT810101","GRAR740102","BEGF750102", "Labels"])
