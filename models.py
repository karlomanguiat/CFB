import pandas  as pd 
import matplotlib.pyplot as plt
import time 
import numpy as np

from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn import datasets

from joblib import dump, load


from sklearn.model_selection import train_test_split

# Change file name to desired dataset BALANCED-TRAINING-DATASET-WX.csv or UB-TRAINING-DATASET-WX.csv on Datasets folder.
# X = window size.

data = pd.read_csv("./Datasets/BALANCED-TRAINING-DATASET-W5.csv") 
X = data[['ZIMJ680101','BHAR880101','HOPT810101','GRAR740102','BEGF750102']]
y = data['Labels']

# split into 70:30 ration 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 50) 
  
# describes info about train and test set 
print("X_train dataset: ", X_train.shape) 
print("y_train dataset: ", y_train.shape) 
print("X_test dataset: ", X_test.shape) 
print("y_test dataset: ", y_test.shape) 
print()

ns_probs = [0 for _ in range(len(y_test))]

# fit a model

print("Developing Naive Bayes....")
model1 = GaussianNB()
print("Fitting Naive Bayes....")
model1.fit(X_train, y_train)
dump(model1, "nb_model.joblib")
print("NB - Saved!")
print()


print("Developing Random Forests....")
model2 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=30, max_features=0.3, max_leaf_nodes=3,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=100,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
print("Fitting Random Forests....")
model2.fit(X_train, y_train)
dump(model2, "rf_model.joblib")
print("RF - Saved!")
print()


n_estimators = 3

print("Developing SVM models....")
model3 = OneVsRestClassifier(BaggingClassifier(LinearSVC(class_weight='balanced', max_iter = 100000), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
print("Fitting SVM models....")
model3.fit(X_train, y_train)
dump(model3, "svm_model.joblib")
print("SVM - Saved!")
print()

# predict probabilities
nb_probs = model1.predict_proba(X_test)
rf_probs = model2.predict_proba(X_test)
svm_probs = model3.decision_function(X_test)

# keep probabilities for the positive outcome only
nb_probs = nb_probs[:, 1]
rf_probs = rf_probs[:, 1]


# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
nb_auc = roc_auc_score(y_test, nb_probs)
rf_auc = roc_auc_score(y_test, rf_probs)
svm_auc = roc_auc_score(y_test, svm_probs)


print("NB - Accuracy: %f" % accuracy_score(y_test, model1.predict(X_test))) 
print("NB - AUC score: %f" % nb_auc)

print("RF - Accuracy: %f" % accuracy_score(y_test, model2.predict(X_test))) 
print("RF - AUC score: %f" % rf_auc)

print("SVM - Accuracy: %f" % accuracy_score(y_test, model3.predict(X_test))) 
print("SVM - AUC score: %f" % svm_auc)

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)


plt.style.use('fivethirtyeight')
plt.figure(figsize = (8, 6))
plt.rcParams['font.size'] = 11
    
# Plot both curves
plt.plot(ns_fpr, ns_tpr, 'blue', label = 'baseline')
plt.plot(nb_fpr, nb_tpr, 'red', label = 'NB')
plt.plot(rf_fpr, rf_tpr, 'green', label = 'RF')
plt.plot(svm_fpr, svm_tpr, 'black', label = 'SVM')



plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
filename = "ROC_CURVES.png" 
plt.savefig(filename)
plt.show()


