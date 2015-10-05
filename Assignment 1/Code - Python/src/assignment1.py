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



def l2_loss(y, w, x, l):
    y_error = y - (x * w.T).T
    y_error = y_error * y_error.T
    l2_error = l * w * w.T

    return y_error + l2_error

def l2gradientDescentStep(y, w, x, l):
    my_guess = x * w.T

    stuff =  ((x * w.T).T - y) * x + l * w

    return stuff



def gradientDescent(y, w, x, l, a, epsilon):
    loss = l2_loss(y, w, x, l)
    print loss
    for i in range(200):
        gradient = l2gradientDescentStep(y, w, x, l)
        w = w - a * gradient
        loss = l2_loss(y, w, x, l)
        print loss

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
    test = gradientDescent(y, w, x, l, 0.001, 1.0)
    #problem1()
    #problem2()
    #problem3()

# Giving Python the main it deserves
if __name__ == "__main__": 
	main()
