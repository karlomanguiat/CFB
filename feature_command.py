import os
import sys
import glob
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

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import numpy as np
import pandas as pd

import csv
import math
import joblib

def addColumnName(df):
	header, count = [], 1

	for i in range(len(df.columns) - 1):
		if(i % 5 == 0): 
			header.append("ZIMJ680101." + str(count))
		elif(i % 5 == 1): 
			header.append("BHAR880101." + str(count))
		elif(i % 5 == 2): 
			header.append("HOPT810101." + str(count))
		elif(i % 5 == 3): 
			header.append("GRAR740102." + str(count))
		elif(i % 5 == 4): 
			header.append("BEGF750102." + str(count))
			count += 1

	header.append("Labels")
	df.columns = header
	return df

def slidingWindow():
	win_size = 3                                                        # Specify the size of the sliding window

	f = open('allMatrix.csv', 'r')                                    	# name of the file to be processed
	re = csv.reader(f, quoting=csv.QUOTE_NONE)                          
	fo = open('allMatrix_w3.csv', 'w')                                    # name of the output file
	wr = csv.writer(fo, quoting=csv.QUOTE_NONE, lineterminator='\n')

	window = []
	labels = []
	ctr = 0
	for row in re:
	    feature_num = len(row) - 1                                      # reads how many cells per row excluding the labels column
	    if(ctr == 0):                                                   # repeats the column header
	        for i in range(0, win_size):
	            window = window + row[:-1]
	        wr.writerow(window)
	        window = []
	        ctr+=1
	    elif(ctr != win_size+1):                                        # initializes the window with the first five residues
	        window = window + row[:-1]
	        labels.append(row[-1])
	        ctr+=1
	    else:                                                           # sliding window part
	        labels.append(row[-1])
	        window.append(labels[math.ceil(win_size/2)-1])              # append the label of the center residue to the window
	        labels = labels[1:]
	        wr.writerow(window)
	        window = window[:-1]                                        
	        window = window[feature_num:len(window)]                    # remove the data of the first residue in the window
	        window = window + row[:-1]                                  # add the data of the next residue to the window

	window.append(labels[math.ceil(win_size/2)-1])
	wr.writerow(window)
	f.close()
	fo.close()

	df = pd.read_csv("allMatrix_w3.csv", skiprows=1)
	df = addColumnName(df)
	df.to_csv("newAllMatrix.csv", index=False)


	if os.path.exists("allMatrix.csv"):
	  os.remove("allMatrix.csv")
	  os.remove("allMatrix_w3.csv")
	else:
	  print("The file does not exist")

def getFastaLength(fastaLength):
	count = 0

	for leng in fastaLength: 
		count += leng

	return count

def initMatrices(conv_seq):
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

	return mat1, mat2, mat3, mat4, mat5

def addLabels(fastaLength, mat1, mat2, mat3, mat4, mat5):
	firstIteration = True
	allMatrix = np.array([])

	for k in range(len(fastaLength)):
		newmat = np.concatenate((mat1[k],mat2[k],mat3[k],mat4[k], mat5[k]))
		newmat1 = (np.transpose(newmat[:,:fastaLength[k]]))

		# RESET LABELS
		label = []

		for i in range(fastaLength[k]):
			label.append(99)

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

	return allMatrix

def toCSV(allMatrix):
	df = pd.DataFrame(allMatrix)
	fileName = "allMatrix.csv"
	df.to_csv(fileName, index=False, header = ["ZIMJ680101", "BHAR880101","HOPT810101","GRAR740102","BEGF750102", "Labels"])


def main():
	testFasta = input("Enter input fasta filename: ")
	algo = input("Enter machine learning algorithm (svm, rf, nb): ")

	# Load the 'xxxxx.fasta' sequence set
	alphasyn_seq = load_fasta_file(testFasta)
	alphasyn_seq1 = load_fasta_file(testFasta)

	# Get array of lengths
	fastaLength, fastaSeq, fastaID = [], [], []

	for seq in alphasyn_seq1: 
		# print(seq)
		fastaLength.append(len(seq.data))
		fastaSeq.append(seq.data)
		fastaID.append(seq.identifier)

	print(fastaSeq[0])

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
	mat1, mat2, mat3, mat4, mat5 = initMatrices(conv_seq)
	allMatrix = addLabels(fastaLength, mat1, mat2, mat3, mat4, mat5)

	toCSV(allMatrix)

	slidingWindow()

	test = pd.read_csv("newAllMatrix.csv")
	test_labels = np.array(test.pop('Labels'))

	my_clf = joblib.load("mnb _model.joblib")
	predictions = my_clf.predict_proba(test)
	# pred = np.argmax(predictions)
	# print(pred)

	# print("Counts of label '1': {}".format(sum(pred == 1))) 
	# print("Counts of label '0': {} \n".format(sum(pred == 0)))

	# print("Accuracy: %f" % accuracy_score(test_labels, predictions))
	# print(confusion_matrix(test_labels, predictions))


main()