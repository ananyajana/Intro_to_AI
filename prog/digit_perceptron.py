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
import time

#Preprocessing training data
start_idx = 0
end_idx = 0
total_train_samples = 5000
x_percent = 100
num_instance = int((total_train_samples * x_percent)/100)

H=28
W=28
start_time = time.time()
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
end_idx = num_instance
Y_train = np.array(Y_train[start_idx:end_idx])

#time taken to preprocess the training data
prep_train_time = time.time() - start_time
print('time taken to preprocess the training data: ')
print(x_percent, 'percent of data took ', prep_train_time)


#Preprocessing test data
############################################################
num_instance1=1000
start_time = time.time()
X_test=[[] for k in range(num_instance1)]
#with open('digitdata/validationimages') as f :
with open('digitdata/testimages') as f :
    
    for k in range(num_instance1):
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
#Y_test=np.loadtxt('digitdata/validationlabels')

#time taken to preprocess the test data
prep_test_time = time.time() - start_time
print('time taken to preprocess the test data: ')
print('data took ', prep_test_time)

################################
(m, n, p) = X_train.shape
print(m)
print(n)
print(p)


def countPlus(sample):
    m,n=sample.shape
    return sum(sum(sample==1))

def countX(sample):
    m,n=sample.shape   
    return sum(sum(sample==2))

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
    #xx=sample.reshape(28*28,1)
    return xx

def createFeatureMatrix (XX):
        
    X_fea=[]   
    for idx, row in enumerate(XX):
        X_fea.append(computeFea(row))
    return np.array(X_fea)

##############################################

start_time = time.time()
X_TrainFea = createFeatureMatrix(X_train)
#X_TestFea = createFeatureMatrix(X_test)


m1, n1 =  X_TrainFea.shape

all_classes, counts=np.unique(Y_train, return_counts=True)
NumClass=len(all_classes)
priors = counts/sum(counts)

num_epochs = 5 
epsilon = 0.2
Weights=np.random.rand(n1,NumClass) # hardcoded
b = np.ones((1, NumClass))
#Weights = np.append(Weights, b)
Weights = np.concatenate((Weights, b))

# add one more column to feature x to account for b. np.ones
#np.array(AAA[Y_trainFea, Y_predFea])
for epch in range(num_epochs):
    print("Starting epoch {0}".format(epch))
    #epsilon=epsilon/10.0
    for idx, row in enumerate(X_TrainFea):
        row = np.append(row, 1)
        Scores=np.dot(row.T, Weights)
        true_label = Y_train[idx]
        
        max_val_idx = np.argmax(Scores)
        if max_val_idx != int(true_label):
            Weights[:, max_val_idx] = Weights[:, max_val_idx] - epsilon*(row.T)
            # increase the score of the correct label
            Weights[:, int(true_label)] = Weights[:, int(true_label)] + epsilon*(row.T)

training_time = time.time() - start_time


###########################################
#Checking the dataset against test data
X_TestFea = createFeatureMatrix(X_test)                    
Y_pred=np.array([-100 for k in range(len(X_TestFea))])
        
for idx, row in enumerate(X_TestFea):
    row = np.append(row, 1)
    Scores=np.dot(row.T, Weights)
    # predicted label is the maximum score
    predicted_class_index=np.argmax(Scores)  
    Y_pred[idx]=all_classes[predicted_class_index]

accuracy = sum(sum([Y_test == Y_pred]))

###############
# Calculating standard deviation
result_arr = np.ones(len(Y_test) - accuracy)
zero_arr = np.zeros(accuracy)
result_arr = np.concatenate((result_arr, zero_arr))

result_mean=np.mean(result_arr)
result_std=np.std(result_arr)

print('################### STATISTICS #####################')
print('Accuracy: ', accuracy, 'out of ', len(Y_test), 'percentage: ', (accuracy/len(Y_test))*100, '%' )

print('time taken to preprocess the training data: ')
print(x_percent, 'percent of data took ', prep_train_time)

print('time taken to preprocess the test data: ')
print('data took ', prep_test_time)

print('time taken to train: ')
print(x_percent, 'percent of data took ', training_time)

print('\n\n\n')
print('-------------nice format------------')
print('--------Digit Recognition with Perceptron----------\n')
print('Number of test data points: ', len(Y_test))
#print('Total number of training points', total_train_samples)
print('Percentage of training data used: ', x_percent, '%')
print('Number of training data points: ', len(Y_train))
print('Accuracy: ', (accuracy/len(Y_test))*100, '%' )
print('Prediction Error: ', (100 - ((accuracy/len(Y_test))*100)), '%' )
print('Training time: ', training_time, 'seconds')
print('standard Deviation', result_std)

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
