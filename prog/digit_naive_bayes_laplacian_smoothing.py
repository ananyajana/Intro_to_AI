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

#Laplacian smoothing factor: already given in code
K = 2.0

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
    #f3=countZero(sample)
    #xx=[f1, f2, f3]
    xx=[f1, f2]
    #xx=sample.reshape(28*28,1)
    #xx=list(xx)
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


############### Preprocessing needed for Laplacian Smoothing



############################################################

m1, n1 =  X_TrainFea.shape

all_classes, counts=np.unique(Y_train, return_counts=True)
NumClass=len(all_classes)
priors = counts/sum(counts)

means=np.zeros((NumClass,n1))
stds = np.zeros((NumClass, n1))
for t in all_classes:
    t=int(t)
    subset=X_TrainFea[Y_train==t]
    for fea in range(n1):
        means[t,fea]=np.mean(subset[:,fea])
        stds[t, fea]=np.std(subset[:,fea])
        
training_time = time.time() - start_time

###########################################
#Checking the dataset against test data
X_TestFea = createFeatureMatrix(X_test)
Y_pred=np.array([-100 for k in range(len(X_TestFea))])

for idx, row in enumerate(X_TestFea):
    Prob=np.ones(NumClass)
    for i,c in enumerate(all_classes):
        prob=np.ones(n1)
        #prob = 1
        for fea in range(n1):
            
            fea_indices = np.where(row[fea] == X_TrainFea[:,fea])[0]
            unique_fea, unique_fea_cnt=np.unique(X_TrainFea[:,fea], return_counts=True)
            for l in range(NumClass):
                label_indices = np.where(l == Y_train)[0]
                label_indices = np.array(label_indices)
                fea_indices_set = set(fea_indices)
                label_indices_set = set(label_indices)
                
                fea_new = set(fea_indices_set).intersection(label_indices_set)
                x_count = len(fea_new)
                label_count = len(label_indices)
                label_Fea = [[]]
                for m in label_indices_set:
                    label_Fea = np.append(label_Fea, X_TrainFea[m])
                #label_Fea = np.array(label_Fea)
                unique_fea_in_label, unique_fea_in_label_cnt=np.unique(label_Fea, return_counts=True)
                total_unique_fea_cnt = len(unique_fea_in_label)
                prob[fea] = (x_count + K)/(label_count + total_unique_fea_cnt*K)
            
            #prob = prob*scipy.stats.norm(means[i, fea], stds[i, fea]).pdf(row[fea])
            Prob[i]=Prob[i] * prob[fea]      
        
    predicted_class_index=np.argmax(Prob)  
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
print('--------Digit Recognition with Naive Bayes----------\n')
print('Number of test data points: ', len(Y_test))
#print('Total number of training points', total_train_samples)
print('Percentage of training data used: ', x_percent, '%')
print('Number of training data points: ', len(Y_train))
print('Accuracy: ', (accuracy/len(Y_test))*100, '%' )
print('Prediction Error: ', (100 - ((accuracy/len(Y_test))*100)), '%' )
print('Training time: ', training_time, 'seconds')
print('standard Deviation', result_std)

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