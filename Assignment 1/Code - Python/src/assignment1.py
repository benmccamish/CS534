import csv
import os
import sys
import numpy as np
from numpy import linalg as LA
import math

trainingData = "../Training/train_p1_15.csv"
testingData = "../Testing/test_p1_15.csv"

def generateVariables(filename):
    matrixTemp = np.genfromtxt(filename,delimiter=",")
    #Collects last column, specified in homework as y
    y = matrixTemp[:,-1]
    np.reshape(y, (matrixTemp.shape[0], 1))
    y = np.matrix(y)
    y = y.transpose()
    #Turns y into column vector, more as a sanity check
    #y = y[:, np.newaxis]
    #Takes all but last column for x
    x = np.matrix(np.delete(matrixTemp, -1, 1))

    return x, y

def generateInitialW(x):
    my_size = x.shape
    return np.matrix(np.zeros(my_size[1]))


def l2_loss(N, y, w, x, l):
	gradientError = 0
	print (y[0] - w.transpose()*x[0])
	print np.dot(w.transpose(),x[0])
	for i in xrange(0,N):
		gradientError += np.pow((y[i] - np.dot(w,x[i]),2) + l*LA.norm(w)
	
	#return gradientError

def l2gradientDescentStep(y, w, x, l):
    return (y - (x * w.T).T) * x + l * w

def gradientDescent(N, y, w, x, l, a, epsilon):
    gradient = 1
    while(abs(gradient) > epsilon):
        gradient = l2_loss(N, y, w, x, l)
        w = w - a*gradient

    return w

def problem1():
	pass

def problem2():
	pass

def problem3():
	pass

def main():
    #starting variables can be changed later to be command line if we want

    x, y = generateVariables(trainingData)
    w = generateInitialW(x)
    N = x.shape[0]
    l = 1
    a = .5
    epsilon = .0001
    #print w
    #print x
    #print y

    #test = l2gradientDescentStep(y, w, x, l)
    testProblem = gradientDescent(N, y, w, x, l, a, epsilon)
    #problem1()
    #problem2()
    #problem3()

# Giving Python the main it deserves
if __name__ == "__main__": 
	main()