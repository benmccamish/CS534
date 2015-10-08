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
    x = np.delete(matrixTemp, -1, 1)
    x = x - x.min(axis=0)

    x = x / x.max(axis=0)
    x[:,0] = 1

    x = np.matrix(x)

    return x, y

def generateInitialW(x):
    my_size = x.shape
    return np.matrix(np.ones((1, my_size[1])))

def gradientDescentSSE(y,w,x,l,N):
	sumError = 0
	for i in range(N):
		test = ((w.T*x[i]) - y[i]).T * ((w.T*x[i]) - y[i])
		sumError += np.multiply(test,x[i])

	return sumError 


def testl2_gradientDescent(y,w,x,l):
	pass


def l2_loss(y, w, x, l):
    y_error = y - (x * w.T)

    y_error = y_error.T * y_error
    l2_error = l * w * w.T
    return y_error + l2_error

def l2gradientDescentStep(y, w, x, l):
    stuff =  ((x * w.T) - y).T * x + l * w
    return stuff

def gradientDescent(y, w, x, l, a, epsilon, N):
    loss = l2_loss(y, w, x, l)
    print loss
    for i in range(N):
        gradient = l2gradientDescentStep(y, w, x, l)
        gradient = gradientDescentSSE(y, w, x, l, N)
        w = w - a * gradient
        loss = l2_loss(y, w, x, l)
        print LA.norm(gradient)
        print loss
        #print gradient

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
    epsilon = .0001

    test = l2gradientDescentStep(y, w, x, l)
    test = gradientDescent(y, w, x, l, .001, 0.0001, N)
    #problem1()
    #problem2()
    #problem3()

# Giving Python the main it deserves
if __name__ == "__main__": 
	main()
