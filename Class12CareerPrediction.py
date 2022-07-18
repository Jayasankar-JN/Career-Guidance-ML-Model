# -*- coding: utf-8 -*-
"""
       #Class 12 prediction

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
data = pd.read_csv("/content/MiniProject- 12 - Sheet1.csv")


#copy the data
c_data = data.copy()

#shuffle Data
c_data = c_data.sample(frac = 1)
c_data.shape

#check for nulls
c_data[c_data.isnull().any(axis= 1)].head()

#dropping nulls
c_data = c_data.dropna()



#deleting age 
del c_data['Age']





#label encoding datas

#replace gender- male with 0 and female with 1
#replace bc12 - state 0 ,cbse 1 ,isc 2
#replace stream - biology science 0 ,computer science 1 ,commerce 2
#replace tuition - yes 1, no 0
#Entrance_Coaching - no 0 ,medicine 1 enggineering 2 law 3
#replace liked_Stream -  engineering 0 medicine 1 pharmacy 2 commerce 3 arts&science 4 nursing 5 management 6 
#replace learning method -practical 0 , theoritical 1
#replace preference - social welfare 0 , personal gain 1
#replace approach - intutive 0 ,concept 1 ,computa 2
#replace preferd job - professional 0 ,non professional 1
#replace intensive research  - yes 1 ,no 0


u_data = c_data.copy()


c_data['Gender'] = c_data['Gender'].replace(['Male','Female'],[0,1])
c_data['BC12'] = c_data['BC12'].replace(['State Government','CBSE','ISC'],[0,1,2])
c_data['Stream'] = c_data['Stream'].replace(['Biology Science','Computer Science','Commerce'],[0,1,2])
c_data['Tuition'] = c_data['Tuition'].replace(['No','Yes'],[0,1])
c_data['Entrance_Coaching'] = c_data['Entrance_Coaching'].replace(['No entrance coaching','Medicine','Engineering', 'Law'],[0,1,2,3])
c_data['Liked_Stream'] = c_data['Liked_Stream'].replace(['Engineering', 'Medicine', 'Pharmacy', 'Commerce',
       'Arts & Science', 'Nursing', 'Management'],[0,1,2,3,4,5,6])
c_data['Selected_Stream'] = c_data['Selected_Stream'].replace(['Engineering', 'Medicine', 'Pharmacy', 'Commerce',
       'Arts & Science', 'Nursing', 'Management',],[0,1,2,3,4,5,6])
c_data['Learning_Method'] = c_data['Learning_Method'].replace(['Practical learning','Theoretical learning'],[0,1])
c_data['Preference'] = c_data['Preference'].replace(['Social welfare','Personal gain'],[0,1])
c_data['Approach'] = c_data['Approach'].replace(['Intuitively','Conceptually','Computationally'],[0,1,2])
c_data['PreferedJob'] = c_data['PreferedJob'].replace(['Professional','Non - professional'],[0,1])
c_data['IntensiveResearch'] = c_data['IntensiveResearch'].replace(['No','Yes'],[0,1])

df = c_data.copy()

#selecting y
y = c_data[['Selected_Stream']].copy()


#feature selection
x = df.copy()
del x['Selected_Stream']
del x['Stream_Satisfication']
del x['Liked_Stream']

del x['Age']

"""# ***Random Forest***"""

#spliting data into test and train
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 324)

#Fitting Random Forest model  with data

random_classifier = RandomForestClassifier()
random_classifier.fit(X_train,y_train)

#Predicting values on test data
y_pred = random_classifier.predict(X_test)

#checking test accuracy
accuracy_score(y_test,y_pred)*100

#confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
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

print(classification_report(y_test,y_pred))

#prediction probability
y_pred_pro = random_classifier.predict_proba(X_test)

#calculation of auc
roc_auc_score(y_test,y_pred_pro,multi_class = "ovr")


# save the model to disk so that it can be resued with out training the model again

filename = 'random_12_model.sav'
pickle.dump(random_classifier, open(filename, 'wb'))

#Here Random Forest classifier gives the maximum accuracy thus it is selected for the career prediction
