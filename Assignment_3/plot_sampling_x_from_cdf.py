
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 06:14:13 2018

@author: ananya
"""

#Author: Ananya Jana
# Program for random sampling from a CDF. 
# We first take random samples U from a uniform distribution [0, 1].
# Then get the x-values from the inverse transform function and plot them
# Take care of th sign of the x-values, if the U value is > 0.5, then x is
# positive, else if U value < 0.5, then x is negative, at U = 0.5, x is 0
 
from math import exp, pow
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

def sign(x):
	signs = []
	for e in x:
		#print("e: ", e)
		if e < 0.1 and e > -0.1:
		#if not e :
			#print("0 ")
			signs.append(0)
		elif e > 0:
			#print("1 ")
			signs.append(1)
		else:
			#print("-1 ")
			signs.append(-1)
	return signs
		

def plot_samples_from_cdf():
	#x = Symbol('x')
	sns.set()	#nice background for the plot
	#plotting the function
	xvals = np.arange(0.0001, 5, 0.2)
	#i = 0
	#for e in xvals:
	#	i += 1
	#print("i: ", i)
	
	zvals = sign(xvals)
	#i = 0
	#for e in zvals:
	#	i += 1np.e
	#	print("e: ", e)
	#print("i: ", i)
	
	#yvals = (1/2) + (np.sqrt(1 - np.exp(-(np.power(xvals,2))/np.pi)))/2
	
	#xvals2 = np.arange(-5, -0.0001, 0.2)
	#yvals2 = (1/2) - (np.sqrt(1 - np.exp(-(np.power(xvals,2))/np.pi)))/2
	#yvals = (1/2) + ((zvals) ** np.sqrt(1 - np.exp(-(np.power(xvals,2))/np.pi)))/2
	#yvals = (1/2) + (sqrt(1 - exp(-(pow(xvals,2))/pi)))/2
	#yvals = (np.sqrt(1 - np.exp(-(np.power(xvals,2))/np.pi)))/2
	
	#xvals = np.arange(-2, 1, 0.01)
	#yvals = (xvals**2 + 2)/3
	
	#yvals = np.cos(xvals)
	#yvals = np.((xvals**2 + 2)/3)
	#yvals = np.g(xvals)
	#plt.plot(xvals, yvals)
	#plt.plot(xvals2, yvals2)
	#plt.show()
	s = np.random.uniform(0,1,100)
	
	zvals3 = []
	for x in s:
		if x < 0.5:
			print('less: ', x)
			zvals3.append(-1)
		if x > 0.5:
			print('more: ', x)
			zvals3.append(1)
	print()
	#count, bins, ignored = plt.hist(s, 15, normed=True)
	#plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
	#plt.show()
	
	#np.log([1, np.e, np.e**2, 0]) natural log function example
	xvals3 = zvals3 * np.sqrt(-np.pi * np.log([1 - (2 * s - 1)**2]))

	#for x, y in zip(s, xvals3):
	#	print('x value', x)
	#	print('y value', y)
		#if x < 0.5:
			#y = -y
			#print(y)
	print()
	#for x in s:
	#	print(x)
	#print()
	for x in xvals3:
		print(x)
	print()
	
	#plt.hist(df_swing['my histogram'])
	plt.xlabel('x values')
	plt.ylabel('frequency of x values')
	plt.title('my histogram')
	#xvals3 = [12, 14, 9, 23, 45, 67 ,89 , 54, 65, 76 ,98, 100]
	plt.hist(xvals3, bins = 50, range = (-5, 5), rwidth = 0.2)
	plt.plot()


plot_samples_from_cdf()
