'''
This automates the repeated training and testing of an SVM using a single dataset.
All results are stored in a CSV file with the same file name as the dataset used with "_results" appended to it.

'''

import pandas  as pd 
import matplotlib.pyplot as plt
import time
import tracemalloc
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split 

from sklearn.svm import LinearSVC, NuSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier

#the number of times an svm is trained and tested
iterations = 20

#dataset to be used to train and test the svm (without the ".csv")
filename = "subset3_w9"

data = pd.read_csv(filename + ".csv") 
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

result = []
for i in range(0, iterations):
    print("Run ", i+1)
    start_time = time.process_time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 50)

    n_estimators = 3

    #print("Developing SVM models....")
    model3 = OneVsRestClassifier(BaggingClassifier(LinearSVC(class_weight='balanced', max_iter = 100000), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
    #print("Fitting SVM models....")
    model3.fit(X_train, y_train)

    svm_probs = model3.decision_function(X_test)
    svm_auc = roc_auc_score(y_test, svm_probs)
    print("SVM - Accuracy: %f" % accuracy_score(y_test, model3.predict(X_test))) 
    print("SVM - AUC score: %f" % svm_auc)
    report = classification_report(y_test, model3.predict(X_test), output_dict=True)
    print(report)
    total_time = time.process_time() - start_time
    print("Time elapsed: " + str(total_time))
    sub_result = [svm_auc, accuracy_score(y_test, model3.predict(X_test)), report['0']['recall'], report['1']['recall'], total_time]

    tracemalloc.start()
    model3 = OneVsRestClassifier(BaggingClassifier(LinearSVC(class_weight='balanced', max_iter = 100000), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
    #print("Fitting SVM models....")
    model3.fit(X_train, y_train)

    svm_probs = model3.decision_function(X_test)
    svm_auc = roc_auc_score(y_test, svm_probs)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}")
    sub_result.append(peak / 10**6)
    result.append(sub_result)
    tracemalloc.stop()

results = pd.DataFrame(result, columns=['AUC', 'Accuracy', 'Specificity', 'Sensitivity', 'Time', 'Memory'])
results.to_csv(filename + "_results.csv", index=False)
