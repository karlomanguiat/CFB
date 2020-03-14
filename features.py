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
	print(fromFile)
	splitArray = fromFile.split(' ', 1)

	# Remove last element from split
	splitArray.pop(0)
	y = splitArray[0].split(' ')
	y = sorted([int(i) for i in y[:-1]])

	return y


# Load the 'xxxxx.fasta' sequence set
alphasyn_seq = load_fasta_file("./Sequences/train157-180.fasta")
alphasyn_seq1 = load_fasta_file("./Sequences/train157-180.fasta")

# Get array of lengths
fastaLength, fastaID, count = [],[], 0

for seq in alphasyn_seq1:
	fastaLength.append(len(seq.data))
	fastaID.append(seq.identifier)

for leng in fastaLength: 
	count += leng


print(fastaLength)

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

# Mean polarity (Radzicka-Wolfenden, 1988)
fs.add(get_aaindex_file("RADA880108"))

# Average accessible surface area (Janin et al., 1978)
fs.add(get_aaindex_file("JANJ780101"))

# Polarity (Zimmerman et al., 1968)
fs.add(get_aaindex_file("ZIMJ680103"))

# Conformational parameter of beta-turn (Beghin-Dirkx, 1975)
fs.add(get_aaindex_file("BEGF750103"))

# Hydrophobicity index (Argos et al., 1982)
fs.add(get_aaindex_file("ARGP820101"))

# Accessible surface area (Radzicka-Wolfenden, 1988)
fs.add(get_aaindex_file("RADA880106"))

# Average accessible surface area (Janin et al., 1978)
fs.add(get_aaindex_file("JANJ780101"))

# Normalized frequency of beta-turn (Chou-Fasman, 1978a)
fs.add(get_aaindex_file("CHOP780101"))

# Flexibility parameter for one rigid neighbors (Karplus-Schulz, 1985)
fs.add(get_aaindex_file("KARP850102"))

# Flexibility parameter for two rigid neighbors (Karplus-Schulz, 1985)
fs.add(get_aaindex_file("KARP850103"))

# Normalized flexibility parameters (B-values) for each residue surrounded by none rigid neighbours (Vihinen et al., 1994)
fs.add(get_aaindex_file("VINM940102"))

# Normalized flexibility parameters (B-values) for each residue surrounded by one rigid neighbours (Vihinen et al., 1994)
fs.add(get_aaindex_file("VINM940103"))

# Normalized flexibility parameters (B-values) for each residue surrounded by two rigid neighbours (Vihinen et al., 1994)
fs.add(get_aaindex_file("VINM940104"))

conv_seq = fs(alphasyn_seq)


# # OPEN FILE AND READING FILE
f = open("./Sequences/training_labels","r")
lines = f.readlines()
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

# Mean polarity (Radzicka-Wolfenden, 1988)
mat11 = np.matrix(columns(conv_seq, feature="RADA880108", transpose=True))

# Average accessible surface area (Janin et al., 1978)
mat12 = np.matrix(columns(conv_seq, feature="JANJ780101", transpose=True))

# Polarity (Zimmerman et al., 1968)
mat13 = np.matrix(columns(conv_seq, feature="ZIMJ680103", transpose=True))

# Conformational parameter of beta-turn (Beghin-Dirkx, 1975)
mat14 = np.matrix(columns(conv_seq, feature="BEGF750103", transpose=True))

# Hydrophobicity index (Argos et al., 1982)
mat15 = np.matrix(columns(conv_seq, feature="ARGP820101", transpose=True))

# Accessible surface area (Radzicka-Wolfenden, 1988)
mat16 = np.matrix(columns(conv_seq, feature="RADA880106", transpose=True))

# Average accessible surface area (Janin et al., 1978)
mat17 = np.matrix(columns(conv_seq, feature="JANJ780101", transpose=True))

# Normalized frequency of beta-turn (Chou-Fasman, 1978a)
mat18 = np.matrix(columns(conv_seq, feature="CHOP780101", transpose=True))

# Flexibility parameter for one rigid neighbors (Karplus-Schulz, 1985)
mat19 = np.matrix(columns(conv_seq, feature="KARP850102", transpose=True))

# Flexibility parameter for two rigid neighbors (Karplus-Schulz, 1985)
mat20 = np.matrix(columns(conv_seq, feature="KARP850103", transpose=True))

# Normalized flexibility parameters (B-values) for each residue surrounded by none rigid neighbours (Vihinen et al., 1994)
mat21 = np.matrix(columns(conv_seq, feature="VINM940102", transpose=True))

# Normalized flexibility parameters (B-values) for each residue surrounded by one rigid neighbours (Vihinen et al., 1994)
mat22 = np.matrix(columns(conv_seq, feature="VINM940103", transpose=True))

# Normalized flexibility parameters (B-values) for each residue surrounded by two rigid neighbours (Vihinen et al., 1994)
mat23 = np.matrix(columns(conv_seq, feature="VINM940104", transpose=True))

# print(mat1, len(mat1), "1")
# print(mat2, len(mat2), "2")
# print(mat3, len(mat3), "3")
# print(mat4, len(mat4), "4")
# print(mat5, len(mat5), "5")

# print(mat6, len(mat6), "6")
# print(mat7, len(mat7), "7")
# print(mat8, len(mat8), "8")
# print(mat9, len(mat9), "9")
# print(mat10, len(mat10), "10")

# print(mat11, len(mat11), "11")
# print(mat12, len(mat12), "12")
# print(mat13, len(mat13), "13")
# print(mat14, len(mat14), "14")
# print(mat15, len(mat15), "15")

# print(mat16, len(mat16), "16")
# print(mat17, len(mat17), "17")
# print(mat18, len(mat18), "18")
# print(mat19, len(mat19), "19")
# print(mat20, len(mat20), "20")

# print(mat21, len(mat21), "21")
# print(mat22, len(mat22), "22")
# print(mat23, len(mat23), "23")

# START LOOP HERE

firstIteration = True
allMatrix = np.array([])
for k in range(len(fastaLength)):
	newmat = np.concatenate((mat1[k], mat2[k], mat3[k], mat4[k], mat5[k], mat6[k], mat7[k], mat8[k], mat9[k], mat10[k], mat11[k], mat12[k], mat13[k], mat14[k], mat15[k], mat16[k], mat17[k], mat18[k], mat19[k], mat20[k], mat21[k], mat22[k], mat23[k]))
	newmat1 = (np.transpose(newmat[:,:fastaLength[k]]))

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

	# Combine two matrices
	finalMatrix = np.append(newmat1, col_vec, axis=1)

	if firstIteration: 
		allMatrix = finalMatrix
		firstIteration = False
	else: allMatrix = np.concatenate((allMatrix, finalMatrix))

	print(k+1)
	print(fastaLength[k])
	print(fastaID[k])
	print("finalmatrix length: %i" % len(finalMatrix))
	print("* * * * * * * * * * * * * * * * * * * * * * * * * * * *\n")


print(allMatrix)
print(count, len(allMatrix))

df = pd.DataFrame(allMatrix)
fileName = "training157-180.csv"
headerName = ["ZIMJ680101","BHAR880101","HOPT810101","GRAR740102","BEGF750102","JOND750101","KARP850101","PRAM900101","KUHL950101","SWER830101","RADA880108","JANJ780101","ZIMJ680103","BEGF750103","ARGP820101","RADA880106","JANJ780101","CHOP780101","VINM940102","KARP850102","KARP850103","VINM940103","VINM940104", "Labels"]
df.to_csv(fileName, index = False, header = headerName)
