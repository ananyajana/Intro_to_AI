
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 06:14:13 2018

@author: ananya
"""

#Author: Ananya Jana
# Program for random sampling from a CDF.
# Get the PDF from CDF by differentiation
# We first take random samples of x from the range of x (-5, 5).
# Then we take random samples U from the uniform distribution [0, 1].
# Then get the P(x = U) from the PDF. If P(x = U) > U, we take that x
# sample x, else discard that sample
 
#from math import exp, pow
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
		
def sign(x):
	sign = 1
	if x < 0:
		sign = -1
	if x == 0:
		sign = 0
	
	return sign
	
def plot_samples_from_pdf():
	x = Symbol('x')
	sns.set()	#nice background for the plot

	samples = 0
	total = 100	# total number of valid x samples that we want to generate
	x_arr = []	# array to hold the sample values of x	
	
	f1 = (1/2) + np.sqrt(1 - np.exp(-(np.power(x,2))/np.pi))/2		#Cumulative Distribution Function F(x) when x > 0
	f1 = (1/2) - np.sqrt(1 - np.exp(-(np.power(x,2))/np.pi))/2		#Cumulative Distribution Function F(x) when x < 0
	f3 = 1/2	#Cumulative Distribution Function F(x) when x = 0
		
	z1 = diff(f1, x, 1)	# Calculating PDF from the CDF by differentiation
	z2 = diff(f2, x, 1)	# Calculating PDF from the CDF by differentiation
	z3 = 0
	
	while samples <= total:
		xval = random.uniform(-5, 5)	# Generate a random sample from the range of x
		z3
		if xval > 0:
			z3 = z1.subs(x, xval)
		if xval < 0:
			z3 = z2.subs(x, xval)
		else:
			z3 = 0
		
		U = np.random.uniform(0,1)
	
		if U <= z3:
			samples = samples + 1
			x_arr.append(xval)
	

	#count, bins, ignored = plt.hist(s, 15, normed=True)
	#plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')

	
	plt.xlabel('x values')
	plt.ylabel('frequency of x values')
	plt.title('my histogram')
	plt.hist(xvals, bins = 50, range = (-5, 5), rwidth = 0.2)
	plt.plot()


plot_samples_from_pdf()
