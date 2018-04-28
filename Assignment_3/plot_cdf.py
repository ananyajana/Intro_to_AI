
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
import numpy as np
import matplotlib.pyplot as plt
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
	
	#plotting the function
	xvals = np.arange(-5, 5, 0.2)
	#i = 0
	#for e in xvals:
	#	i += 1
	#print("i: ", i)
	
	zvals = sign(xvals)
	#i = 0
	#for e in zvals:
	#	i += 1
	#	print("e: ", e)
	#print("i: ", i)
	
	yvals = (1/2) + ((zvals) ** np.sqrt(1 - np.exp(-(np.power(xvals,2))/np.pi)))/2
	#yvals = (1/2) + (sqrt(1 - exp(-(pow(xvals,2))/pi)))/2
	#yvals = (np.sqrt(1 - np.exp(-(np.power(xvals,2))/np.pi)))/2
	
	#xvals = np.arange(-2, 1, 0.01)
	#yvals = (xvals**2 + 2)/3
	
	#yvals = np.cos(xvals)
	#yvals = np.((xvals**2 + 2)/3)
	#yvals = np.g(xvals)
	plt.plot(xvals, yvals)
	plt.show()
	
	



plot_cdf()
