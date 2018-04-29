#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 06:14:13 2018
@author: ananya

This program plots the given CDF and PDF of x w.r.t x.
The question asks to sample x when the CDF is given.

We know that for large enough sample, the values of x will
be distributed according to the PDF of x. Hence, first we
calculate the PDF of x from the CDF by differentiation.
Then we sample x.
"""

#Author: Ananya Jana
# Program for random sampling from a CDF
from math import exp, pow
from sympy import *
#import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from random import *
import time
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
		

def plot_cdf():
	#x = Symbol('x')
	sns.set()	#nice background for the plot
	#plotting the function
	xvals = np.arange(0.00000001, 5, 0.2)
	yvals = (1/2) + (np.sqrt(1 - np.exp(-(np.power(xvals,2))/np.pi)))/2			#Cumultive Distribution Function when x > 0
	
	xvals2 = np.arange(-5, -0.0000001, 0.2)
	yvals2 = (1/2) - (np.sqrt(1 - np.exp(-(np.power(xvals,2))/np.pi)))/2			#Cumultive Distribution Function when x < 0
	
	plt.plot(xvals, yvals)
	plt.plot(xvals2, yvals2)
	plt.show()
	time.sleep(10)
	plt.gcf().clear()	
	

def plot_pdf():
	sns.set()	#nice background for the plot
	#plotting the function
	xvals = np.arange(-5, 5, 0.2)
	
	#Probability Distribution Function when x > 0
	yvals = (np.absolute(xvals)/np.pi)*(1/(np.sqrt(1 - np.exp(-(np.power(xvals,2))/np.pi))))*np.exp(-(np.power(xvals,2))/np.pi)	
	plt.plot(xvals, yvals)
	plt.show()
	time.sleep(10)
	plt.gcf().clear()	

#This function needs to be debugged
def plot_pdf1():
	x = Symbol('x')
	sns.set()	#nice background for the plot
	
	f1 = (1/2) + np.sqrt(1 - np.exp(-(np.power(x,2))/np.pi))/2		#Cumulative Distribution Function F(x) when x > 0
	f1 = (1/2) - np.sqrt(1 - np.exp(-(np.power(x,2))/np.pi))/2		#Cumulative Distribution Function F(x) when x < 0
	f3 = 1/2	#Cumulative Distribution Function F(x) when x = 0
		
	z1 = diff(f1, x, 1)	# Calculating PDF from the CDF by differentiation
	z2 = diff(f2, x, 1)	# Calculating PDF from the CDF by differentiation

	xvals = np.arange(0.00000001, 5, 0.2)
	yvals = z1.subs(x, xval)			#Cumultive Distribution Function when x > 0
	
	xvals2 = np.arange(-5, -0.0000001, 0.2)
	yvals2 = z2.subs(x, xvals2)
	
	plt.plot(xvals, yvals)
	plt.plot(xvals2, yvals2)
	plt.show()
	
	
def plot_samples_from_pdf():
	sns.set()	#nice background for the plot

	samples = 0
	total = 1000	# total number of valid x samples that we want to generate
	x_arr = []	# array to hold the sample values of x	
	
	while samples <= total:
		xvals = np.random.uniform(-5, 5)	# Generate a random sample from the range of x
		yvals = (np.absolute(xvals)/np.pi)*(1/(np.sqrt(1 - np.exp(-(np.power(xvals,2))/np.pi))))*np.exp(-(np.power(xvals,2))/np.pi)
		
		U = np.random.uniform(0,1)
	
		if U <= yvals:
			samples = samples + 1
			x_arr.append(xvals)
	

	#count, bins, ignored = plt.hist(s, 15, normed=True)
	#plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')

	
	plt.xlabel('x values')
	plt.ylabel('frequency of x values')
	plt.title('my histogram')
	plt.hist(x_arr, bins = 50, range = (-5, 5), rwidth = 0.8)
	plt.plot()



plot_cdf()
plot_pdf()
plot_samples_from_pdf()