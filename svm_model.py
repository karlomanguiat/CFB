import pandas  as pd 
import matplotlib.pyplot as plt
import time 
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split 

from sklearn.svm import LinearSVC, NuSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier

# nu value used for the SVC
nu = 0.01

data = pd.read_csv("subset1_w3.csv") 
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 50)

n_estimators = 3

print("Developing SVM models....")
model3 = OneVsRestClassifier(BaggingClassifier(NuSVC(nu = nu, max_iter = 100000), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
print("Fitting SVM models....")
model3.fit(X_train, y_train)

svm_probs = model3.decision_function(X_test)
svm_auc = roc_auc_score(y_test, svm_probs)
print("SVM - Accuracy: %f" % accuracy_score(y_test, model3.predict(X_test))) 
print("SVM - AUC score: %f" % svm_auc)
print(classification_report(y_test, model3.predict(X_test)))

base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)

plt.style.use('fivethirtyeight')
plt.figure(figsize = (8, 6))
plt.rcParams['font.size'] = 11
plt.plot(base_fpr, base_tpr, 'red', label = 'baseline')
plt.plot(svm_fpr, svm_tpr, 'blue', label = 'SVM')

plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve');
filename = "ROC_CURVES.png" 
plt.savefig(filename)
plt.show()
