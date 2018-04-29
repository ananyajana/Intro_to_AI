#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 06:14:13 2018
@author: ananya
"""

#Author: Ananya Jana
# Program for random sampling from a CDF
from math import exp, pow
from sympy import *
#import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
	yvals2 = (1/2) - (np.sqrt(1 - np.exp(-(np.power(xvals,2))/np.pi)))/2
	
	#yvals = (1/2) + ((zvals) ** np.sqrt(1 - np.exp(-(np.power(xvals,2))/np.pi)))/2		#Cumulative Distribution Function F(x) when x < 0
	
	#yvals = (1/2) + (sqrt(1 - exp(-(pow(xvals,2))/pi)))/2
	#yvals = (np.sqrt(1 - np.exp(-(np.power(xvals,2))/np.pi)))/2
	
	#xvals = np.arange(-2, 1, 0.01)
	#yvals = (xvals**2 + 2)/3
	
	#yvals = np.cos(xvals)
	#yvals = np.((xvals**2 + 2)/3)
	#yvals = np.g(xvals)
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

plot_cdf()
plot_pdf()