#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 05:08:16 2018

@author: ananya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

num_instance=5000
H=28
W=28

X_train=[[] for k in range(num_instance)]
with open('digitdata/trainingimages') as f :
    
    for k in range(num_instance):
        #count_hash = 0
        #count_plus = 0
        datainstance=[[] for k in range(H)]
        for j in range(H):
            line=f.readline()
            l=[]
            for ch in range(len(line)-1):
                
                ## if you try to convert to numbers 
                ## do it here
                if line[ch] == ' ':
                    l.append(0)
                if line[ch] == '+':
                    l.append(1)
                    #count_plus += 1
                if line[ch] == '#':
                   l.append(2)
                   #count_hash += 1
                   
            datainstance[j]=l
            print(line)
        
        #print(count_hash)
        #print(count_plus)
        print("___________________________________")
        X_train[k]=datainstance
        

X_train=np.array(X_train)
# save as numpy format

# read train targets
Y_train=np.loadtxt('digitdata/traininglabels')





num_instance=1000
X_test=[[] for k in range(num_instance)]
with open('digitdata/testimages') as f :
    
    for k in range(num_instance):
        #count_hash = 0
        #count_plus = 0
        datainstance=[[] for k in range(H)]
        for j in range(H):
            line=f.readline()
            l=[]
            for ch in range(len(line)-1):
                
                ## if you try to convert to numers 
                ## do it here
                if line[ch] == ' ':
                    l.append(0)
                if line[ch] == '+':
                    l.append(1)
                    #count_plus += 1
                if line[ch] == '#':
                   l.append(2)
                   #count_hash += 1
                   
            datainstance[j]=l
            print(line)
        #print(count_hash)
        #print(count_plus)
        print("___________________________________")
        X_test[k]=datainstance
        

X_test=np.array(X_test)
Y_test=np.loadtxt('digitdata/testlabels')

################################
(m, n, p) = X_train.shape
print(m)
print(n)
print(p)


def countPlus(sample):
    m,n=sample.shape
    return sum(sum(sample==2))

def countX(sample):
    m,n=sample.shape   
    return sum(sum(sample==1))

def countZero(sample):
    m,n=sample.shape   
    return sum(sum(sample==0))



def computeFea(sample):
    m,n=sample.shape
    f1=countX(sample)
    f2=countPlus(sample)
    f3=countZero(sample)
    #xx=[f1, f2, f3]
    xx=[f1, f2]
    return xx

def createFeatureMatrix (XX):
        
    X_fea=[]   
    for idx, row in enumerate(XX):
        X_fea.append(computeFea(row))
    return np.array(X_fea)

##############################################



X_TrainFea = createFeatureMatrix(X_train)
X_TestFea = createFeatureMatrix(X_test)


m1, n1 =  X_TrainFea.shape

all_classes, counts=np.unique(Y_train, return_counts=True)
NumClass=len(all_classes)
priors = counts/sum(counts)

num_epochs = 5 
epsilon = 0.5
Weights=np.random.rand(n1,NumClass) # hardcoded

# add one more column to feature x to account for b. np.ones
#np.array(AAA[Y_trainFea, Y_predFea])
for epch in range(num_epochs):
    print("Starting epoch {0}".format(epch))
    epsilon=epsilon/10.0
    for idx, row in enumerate(X_TrainFea):
        Scores=np.dot(row.T, Weights)
        true_label = Y_train[idx]
        
        for i, c in enumerate(all_classes):
            if c == true_label:
                if Scores[i] < 0:
                    Weights[:, i] = Weights[:, i] - epsilon*(row.T)
            else:
                if Scores[i] > 0:
                    Weights[:, i] = Weights[:, i] - epsilon*(row.T)
                    
Y_pred=np.array([-100 for k in range(len(X_TestFea))])
        
for idx, row in enumerate(X_TestFea):
    Scores=np.dot(row.T, Weights)
    # predicted label is the maximum score
    predicted_class_index=np.argmax(Scores)  
    Y_pred[idx]=all_classes[predicted_class_index]

accuracy = sum(sum([Y_test == Y_pred]))
print('Accuracy: ', accuracy, 'out of ', len(Y_test), 'percentage: ', (accuracy/len(Y_test))*100, '%' )

"""        
means=np.zeros((NumClass,n1))
stds = np.zeros((NumClass, n1))
for t in all_classes:
    t=int(t)
    subset=X_TrainFea[Y_train==t]
    for fea in range(n1):
        means[t,fea]=np.mean(subset[:,fea])
        stds[t, fea]=np.std(subset[:,fea])
        

Y_pred=np.array([-100 for k in range(len(X_TestFea))])

for idx, row in enumerate(X_TestFea):
    Prob=np.zeros(NumClass)
    for i,c in enumerate(all_classes):
        prob = 1
        for fea in range(n1):
            prob = prob*scipy.stats.norm(means[i, fea], stds[i, fea]).pdf(row[fea])
            Prob[i]=prob
        
    predicted_class_index=np.argmax(Prob)  
    Y_pred[idx]=all_classes[predicted_class_index]
    
accuracy = sum(sum([Y_test == Y_pred]))
print('Accuracy: ', accuracy, 'out of ', len(Y_test), 'percentage: ', (accuracy/len(Y_test))*100, '%' )
"""
# another way of counting plus and hash

"""
count_x1 = 0
count_x2 = 0

for i in range(H):
    for j in range(W):
        #print(X_test[0, i, j])
        if X_train[1, i, j] == 1:
            count_x1 += 1
        if X_train[1, i, j] == 2:
            count_x2 += 1
print(count_x1)
print(count_x2)
"""
