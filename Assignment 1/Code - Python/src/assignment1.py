import csv
import os
import sys
import numpy as np
import math

trainingData = "../Training/train_p1_15.csv"
testingData = "../Testing/test_p1_15.csv"

def generateVariables(filename):
	matrixTemp = np.genfromtxt(filename,delimiter=",")
	#Collects last column, specified in homework as y
	y = matrixTemp[:,-1]
	#Turns y into column vector, more as a sanity check
	y = y[:, np.newaxis]
	#Takes all but last column for x
	x = np.delete(matrixTemp, -1, 1)

	return x, y

def generateInitialW():
	pass

def gradientDescentStep(N, y, w, x, l):
	gradientError = 0
	for i in xrange(0,N):
		gradientError += math.pow((y[i] - w.transpose()*x[i]),2) + l*math.pow(abs(w),2)

def gradientDescent(N, y, w, x, l, epsilon):
	gradientError = 1
	while(gradientError > epsilon):
		gradientError = gradientDescentStep(N, y, w, x, l)
		w = w - l*gradientError

	return w

def problem1():
	pass

def problem2():
	pass

def problem3():
	pass

def main():
	#starting variables can be changed later to be command line if we want
	w = generateInitialW()
	x, y = generateVariables(trainingData)
	N = len(x[0])
	l = 1
	epsilon = .0001

	#problem1()
	#problem2()
	#problem3()

# Giving Python the main it deserves
if __name__ == "__main__": 
	main()