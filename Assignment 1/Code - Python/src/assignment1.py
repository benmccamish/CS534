import csv
import os
import sys
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_pdf import PdfPages

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

def gradientDescentTesting(y,w,x,l,N):
	lossList = list()
	gradientList = list()
	for i in range(N):
		loss = l2_loss(y[i], w, x[i], l)
		gradient = l2gradientDescentStep(y[i], w, x[i], l)
		lossList.append(loss.item(0))
		gradientList.append(LA.norm(gradient))
		#w = w - a * gradient
		print LA.norm(gradient)
		print loss.item(0)

	return lossList, gradientList, w


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
    lossList = list()
    gradientList = list()
    for i in range(N):
        gradient = l2gradientDescentStep(y, w, x, l)
        lossList.append(loss.item(0))
        gradientList.append(LA.norm(gradient))
        w = w - a * gradient
        loss = l2_loss(y, w, x, l)
        print LA.norm(gradient)
        print loss.item(0)

    return lossList, gradientList, w

def problem1TrainandTest(y,w,x,l,a,epsilon,N,fileName):
	#plt.clf()
	
	trainingLoss, trainingGradient, w = gradientDescent(y, w, x, l, a, epsilon, N)
	x, y = generateVariables(testingData)
	testingLoss, testingGradient, w = gradientDescentTesting(y, w, x, l, N)
	
	plt.figure(0)
	plt.plot(trainingLoss, label='a='+str(a))
	
	plt.figure(1)
	plt.plot(trainingGradient, label='a='+str(a))

	plt.figure(2)
	plt.plot(testingGradient, label='a='+str(a))
	
	plt.figure(3)
	plt.plot(testingLoss, label='a='+str(a))


def problem1(y, w, x, l, a, epsilon, N):
	problem1TrainandTest(y,w,x,l,0.00001,epsilon,N, 'a00001')
	problem1TrainandTest(y,w,x,l,0.0001,epsilon,N, 'a0001')
	problem1TrainandTest(y,w,x,l,0.001,epsilon,N, 'a001')
	problem1TrainandTest(y,w,x,l,0.01,epsilon,N, 'a01')
	problem1TrainandTest(y,w,x,l,0.1,epsilon,N, 'a1')

	pp = PdfPages('trainingLoss.pdf')
	plt.figure(0)
	plt.ylim((0,60000))
	plt.xlim((-5,100))
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.savefig(pp, format='pdf',bbox_inches = 'tight')
	pp.close()
	#plt.clf()

	pp = PdfPages('trainingGradient.pdf')
	plt.figure(1)
	plt.ylim((0,10000))
	plt.xlim((-5,100))
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.savefig(pp, format='pdf',bbox_inches = 'tight')
	pp.close()

	pp = PdfPages('testingGradient.pdf')
	plt.figure(2)
	plt.ylim((0,1000))
	plt.xlim((-5,100))
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.savefig(pp, format='pdf',bbox_inches = 'tight')
	pp.close()

	pp = PdfPages('testingLoss.pdf')
	plt.figure(3)
	plt.ylim((0,5000))
	plt.xlim((-5,100))
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.savefig(pp, format='pdf',bbox_inches = 'tight')
	pp.close()


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
    a = 0.001
    epsilon = .0001

    #test = l2gradientDescentStep(y, w, x, l)
    #test = gradientDescent(y, w, x, l, 0.001, 0.0001, N)

    #gradientDescentSSE(y, w, x, l, N)
    problem1(y, w, x, l, a, epsilon, N)
    #problem1TrainandTest(y,w,x,l,0.001,epsilon,N, 'a001')
    #plt.show()
    #problem2()
    #problem3()

# Giving Python the main it deserves
if __name__ == "__main__": 
	main()
