#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:05:32 2019

@author: saianuroopkesanapalli
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:53:16 2019

@author: ani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 19:42:02 2019
@author: mandeep
"""

# import numpy, pandas and time
import numpy as np
import pandas as pd
import time

# visual libraries
from matplotlib import pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
plt.style.use('ggplot')

# sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,matthews_corrcoef,classification_report,roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import confusion_matrix

# Read the data in the CSV file using pandas
df = pd.read_csv('creditcard.csv')
df.isnull().any().sum()
fraud = df[df['Class'] == 1]
nonFraud = df[df['Class'] == 0]


# Division of dataset into bags
print("Creating " + str(len(nonFraud)//492) + " bags")
print("--------------------------")
total_list = []
# 1st bag
nf1 = pd.DataFrame(nonFraud[0:492]) 
nf2 = pd.DataFrame(nonFraud[492:492 + 100])
nf3 = pd.concat([nf1,nf2,fraud])
nf3 = nf3.sample(frac = 1)
total_list.append(nf3)

# bags in the middle
for x in range(1,(len(nonFraud)//492)):
    nf1 = pd.DataFrame(nonFraud[(x*492) - 50 :x*492])
    nf2 = pd.DataFrame(nonFraud[x*492:(x+1)*492])
    nf3 = pd.DataFrame(nonFraud[(x+1)*492:(x+1)*492 + 50])
    nf3 = pd.concat([nf1,nf2,nf3,fraud])
    nf3 = nf3.sample(frac = 1)
    total_list.append(nf3)

# last bag
nf1 = pd.DataFrame(nonFraud[(577*492) - 100:577*492])
nf2 = pd.DataFrame(nonFraud[577*492:])
nf3 = pd.concat([nf1,nf2,fraud])
nf3 = nf3.sample(frac = 1)
total_list.append(nf3)

# weights for false negative and false positive in the cost function
fn = 0
fp = 1
high_acc = 0
low_acc = 1
cw = {}
cw[0]=fn
cw[1]=fp

max_prec=0
max_rec=0
max_acc=0
max_fn=0
max_fb=1

print("Determining Optimal weight(fn,fp)")
print("-----------------------------------")
# checking for optimal value of weight(fn,fp) in steps of 0.05
for y in range(1,20):
    fn = fn + 0.05
    fp = 1 - fn
    cw[0]=fn
    cw[1]=fp
    tot_prec=0
    tot_rec=0
    tot_acc=0
    bagno=0
    print("weight(fn,fp) = ("+str(fn)+","+str(fp)+")")
    # calculating the sums of accuracy, precision and recall for all bags at the given value of weight(fn,fp)
    for x in total_list:
        print("weight("+str(fn)+","+str(fp)+")->"+"Bag "+str(bagno))
        features = x.drop(['Class'], axis = 1)
        labels = pd.DataFrame(x['Class'])
        features = features.reset_index()
        features = features.drop(['index'], axis = 1)
        labels = labels.reset_index()
        labels = labels.drop(['index'], axis = 1)
        features_array = features.values
        labels_array = labels.values
        X_train,X_test,y_train,y_test = train_test_split(features_array,labels_array,test_size=0.20)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        classifier = svm.SVC(class_weight=cw,kernel='rbf',gamma = "scale")
        y_train = y_train.ravel()
        classifier.fit(X_train,y_train)
        predicted = classifier.predict(X_test)
        a_accuracy_score = accuracy_score(y_test,predicted)
        a_precison_score = precision_score(y_test,predicted)
        a_recall_score = recall_score(y_test,predicted)
        a_f1_score = f1_score(y_test,predicted)
        a_MCC = matthews_corrcoef(y_test,predicted)
        tot_acc = tot_acc + a_accuracy_score
        tot_prec = tot_prec + a_precison_score
        tot_rec = tot_rec + a_recall_score
        
        if(a_accuracy_score > high_acc):
            high_acc = a_accuracy_score
        if(low_acc > a_accuracy_score):
            low_acc = a_accuracy_score
        bagno=bagno+1
    # calculating the average accuracy, precision and recall of all the bags at the given value of weight(fn,fp)  
    tot_acc= tot_acc / (len(nonFraud)//492)
    tot_prec = tot_prec / (len(nonFraud)//492)
    tot_rec = tot_rec / (len(nonFraud)//492)
    tot_tot=(tot_acc+tot_prec+tot_rec)/3
    # calculating the average score of classifiers at the given value of weight(fn,fp)  
    tot_max=(max_acc+max_prec+max_rec)/3
    # updating the value of weight(fn,fp) at which maximum average score of classifiers is achieved
    if(tot_tot>tot_max):
        max_fn=fn
        max_fp=fp
        max_acc=tot_acc
        max_prec=tot_prec
        max_rec=tot_rec
print("Optimal weight(fn,fp) = "+str(max_fn)+" "+str(max_fp))

print("Determining bag weights and pooling unused datasets")
print("----------------------------------------------------")
# assigning bag weights on the basis of the sum of accuracy, precision and recall of each bag at Optimal weight(fn,fp)
tot_acc = 0
tot_prec = 0
tot_rec = 0
high_acc = 0
low_acc = 1
arg_max_bag_weight = 0
bag_weights = []
classifier_bag = []
testing_feature_dataframe = []
testing_dataframe = []
i = 1
for x in total_list:
    cw[0]=max_fn
    cw[1]=max_fp
    features = x.drop(['Class'], axis = 1)
    labels = pd.DataFrame(x['Class'])
    features = features.reset_index()
    features = features.drop(['index'], axis = 1)
    labels = labels.reset_index()
    labels = labels.drop(['index'], axis = 1)
    features_array = features.values
    labels_array = labels.values
    X_train,X_test,y_train,y_test = train_test_split(features_array,labels_array,test_size=0.20)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # pooling the test datasets (each of which is 20% of the bag size) of each bag
    if i ==1:
        testing_feature_dataframe = pd.DataFrame(X_test[:,:])
        testing_dataframe = pd.DataFrame(y_test[:,:])
    else:
        testing_feature_dataframe = pd.concat([testing_feature_dataframe,pd.DataFrame(X_test[:,:])])
        testing_dataframe = pd.concat([testing_dataframe,pd.DataFrame(y_test[:,:])])    
    classifier = svm.SVC(C=5,class_weight=cw,kernel='rbf',gamma = "scale")
    y_train = y_train.ravel()
    classifier.fit(X_train,y_train)
    classifier_bag.append(classifier)
    predicted = classifier.predict(X_test)
    a_accuracy_score = accuracy_score(y_test,predicted)
    a_precison_score = precision_score(y_test,predicted)
    a_recall_score = recall_score(y_test,predicted)
    bag_weights.append((a_accuracy_score + a_precison_score + a_recall_score)/3)
    a_f1_score = f1_score(y_test,predicted)
    a_MCC = matthews_corrcoef(y_test,predicted)
    tot_acc = tot_acc + a_accuracy_score
    tot_prec = tot_prec + a_precison_score
    tot_rec = tot_rec + a_recall_score
    i += 1
# finding the bag which has the maximum weight at optimal values of weight(fn,fp)
arg_max_bag_weight = bag_weights.index(max(bag_weights))        
bag_weights_average = [x/sum(bag_weights) for x in bag_weights]

# Testing on entire dataset
print("Testing on entire dataset")
print("--------------------------")
features = df.drop(['Class'], axis = 1)
labels = pd.DataFrame(df['Class'])
features = features.reset_index()
features = features.drop(['index'], axis = 1)
labels = labels.reset_index()
labels = labels.drop(['index'], axis = 1)
features_array = features.values
labels_array = labels.values
# Training on 80% of the entire dataset and testing on 20% of the entire dataset
X_train,X_test,y_train,y_test = train_test_split(features_array,labels_array,test_size=0.20)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = svm.SVC(C=5,class_weight=cw,kernel='rbf',gamma = "scale")
y_train = y_train.ravel()
classifier.fit(X_train,y_train)
predicted = classifier.predict(X_test)
a_accuracy_score = accuracy_score(y_test,predicted)
a_precison_score = precision_score(y_test,predicted)
a_recall_score = recall_score(y_test,predicted)    
print("Accuracy: " + str(a_accuracy_score))
print("Precision: " + str(a_precison_score))
print("Recall: " + str(a_recall_score))

  
# Testing on pooled dataset using the classifier of that bag which has the maximum weight at optimal values of weight(fn,fp)
print("Testing on pooled dataset")
print("--------------------------")
features = testing_feature_dataframe
labels = testing_dataframe
features = features.reset_index()
features = features.drop(['index'], axis = 1)
labels = labels.reset_index()
labels = labels.drop(['index'], axis = 1)
features_array = features.values
labels_array = labels.values
sc = StandardScaler()
sc.fit_transform(features_array)
sc.transform(features_array)
predicted = classifier_bag[arg_max_bag_weight].predict(features_array)
a_accuracy_score = accuracy_score(labels_array,predicted)
a_precison_score = precision_score(labels_array,predicted)
a_recall_score = recall_score(labels_array,predicted)
print("Accuracy: " + str(a_accuracy_score))
print("Precision: " + str(a_precison_score))
print("Recall: " + str(a_recall_score))