# -*- coding: utf-8 -*-
"""   
    ### ***Class 10 Prediction***
"""

#importing the required packages 

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score , confusion_matrix,roc_curve,auc,roc_auc_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#import data
data = pd.read_csv("/content/Copy of Mini Project _Class 10  - Sheet1.csv")

#print the columns
data.columns

#copy the data
c_data = data.copy()

#shuffle data
c_data = c_data.sample(frac = 1)

#check for nulls
c_data[c_data.isnull().any(axis= 1)].head()

#dropping nulls
c_data = c_data.dropna()

clean_data = c_data.copy()


#deleting age and cle as it is not relevant for the model

del c_data['Age']
del c_data["CLE"]


#label encoding datas

#replace gender- male  0 , female  1
#replace bc10 - state 0 , cbse 1, icse 2
#replace bc12 - state 0 , cbse 1 , icse 2
#replace tuition - yes 1 ,no 0
#replace liked12 - cs 1, bio 2 ,hum 3, comm 4
#replace selected12 - cs 1, bio 2 ,hum 3, comm 4
#replace learning method -practical 0 , theoritical 1
#replace preference - social welfare 0 , personal gain 1
#replace approach - intutive 0 ,concept 1 ,computa 2
#replace preferd job - professional 0 ,non professional 1
#replace intensive research  - yes 1 ,no 0


c_data['Gender'] = c_data['Gender'].replace(['Male','Female'],[0,1])
c_data['BC10'] = c_data['BC10'].replace(['State Government','CBSE','ICSE'],[0,1,2])
c_data['BC12'] = c_data['BC12'].replace(['State Government','CBSE','ISC'],[0,1,2])
c_data['Tuition'] = c_data['Tuition'].replace(['No','Yes'],[0,1])
c_data['Liked_Stream12'] = c_data['Liked_Stream12'].replace(['Computer Science','Biology Science','Humanities','Commerce'],[1,2,3,4])
c_data['Selected_stream12'] = c_data['Selected_stream12'].replace(['Computer Science','Biology Science','Humanities','Commerce'],[1,2,3,4])
c_data['Learning_Method'] = c_data['Learning_Method'].replace(['Practical learning','Theoretical learning'],[0,1])
c_data['Preference'] = c_data['Preference'].replace(['Social welfare','Personal gain'],[0,1])
c_data['Approach'] = c_data['Approach'].replace(['Intuitively','Conceptually','Computationally'],[0,1,2])
c_data['PreferedJob'] = c_data['PreferedJob'].replace(['Professional','Non - professional'],[0,1])
c_data['IntensiveResearch'] = c_data['IntensiveResearch'].replace(['No','Yes'],[0,1])



df = c_data.copy()

#selecting y
y = c_data[['Selected_stream12']].copy()


#feature selection
x = df.copy()
#removing features that are not required as input

del x['Selected_stream12']
del x['Stream_Satisfication']
del x['Liked_Stream12']



"""### **Model Training**

1.*Decision Tree*
"""

#spliting data into test and train

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 324)

#Fitting decision tree model on data

career_classifier = DecisionTreeClassifier(criterion = 'gini',max_leaf_nodes =12,random_state = 0)
career_classifier.fit(X_train,y_train)

#Predicting values on test data

y_predicted = career_classifier.predict(X_test)

#checking test accuracy

accuracy_score(y_test,y_predicted)*100

#confusion matrix

cnf_matrix = confusion_matrix(y_test,y_predicted)
print(cnf_matrix)

#Classification performance calculation

FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(cnf_matrix)
TN = cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
#F1 Score 2 * (Precision * Recall) / (Precision + Recall)
F1 = 2 * (PPV * TPR) / (PPV + TPR)

print("Accuracy:",ACC)
print("Precision:",PPV)
print("Sensitivity:",TPR)
print("Specificity:",TNR)
print("F1 Score:",F1)

#Classification report generation

print(classification_report(y_test,y_predicted))

# generating the roc curve for classes
fpr = {}
tpr = {}
thresh ={}

n_class = 4

for i in range(1,n_class+1):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_predicted, pos_label=i)
    
# plotting    

plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Computer Science vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Biology Science vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--',color='red', label='Hummanities vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--',color='orange', label='Commerce vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')

#prediction probability
y_pred_pro_dt = career_classifier.predict_proba(X_test)

#calculation of auc

roc_auc_score(y_test,y_pred_pro_dt,multi_class = "ovr")



"""2.Random *Forest*"""

#spliting data into test and train
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 324)

#Fitting Random Forest model  with data

random_classifier = RandomForestClassifier()
random_classifier.fit(X_train,y_train)

#Predicting values on test data

y_pred = random_classifier.predict(X_test)

#checking test accuracy

accuracy_score(y_test,y_pred)*100

#confusion matrix
RF_cnf_matrix = confusion_matrix(y_test,y_pred)
print(cnf_matrix)

#Classification performance calculation
FP = RF_cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
FN = RF_cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
TP = np.diag(RF_cnf_matrix)
TN = RF_cnf_matrix.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
#F1 Score 2 * (Precision * Recall) / (Precision + Recall)
F1 = 2 * (PPV * TPR) / (PPV + TPR)

print("Accuracy:",ACC)
print("Precision:",PPV)
print("Sensitivity:",TPR)
print("Specificity:",TNR)
print("F1 Score:",F1)

# roc curve for classes
fpr = {}
tpr = {}
thresh ={}

n_class = 4

for i in range(1,n_class+1):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred, pos_label=i)
    
# plotting    
plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Computer Science vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Biology Science vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--',color='red', label='Hummanities vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--',color='orange', label='Commerce vs Rest')
plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')

#prediction probability
y_pred_pro_rf = random_classifier.predict_proba(X_test)

#calculation of auc
roc_auc_score(y_test,y_pred_pro_rf,multi_class = "ovr")


# save the model to disk so that it can be resued with out training the model again


filename = 'random_model.sav'
pickle.dump(random_classifier, open(filename, 'wb'))

#Here Random Forest classifier gives the maximum accuracy thus it is selected for the career prediction




