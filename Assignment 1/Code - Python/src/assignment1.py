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

    SSE = lossSSE(y, w, x, l)
    return lossList, gradientList, w, SSE.item(0)

def lossSSE(y, w, x, l):
    y_error = y - (x * w.T)
    y_error = y_error.T * y_error
    return y_error

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
    stopping = 50
    #for i in range(3000):
    while stopping > 1:
        gradient = l2gradientDescentStep(y, w, x, l)
        lossList.append(loss.item(0))
        gradientList.append(LA.norm(gradient))
        w = w - a * gradient
        loss = l2_loss(y, w, x, l)
        print LA.norm(gradient)
        stopping = LA.norm(gradient)
        print loss.item(0)

    SSE = lossSSE(y, w, x, l)
    return lossList, gradientList, w, SSE.item(0)

def problem1TrainandTest(y,w,x,l,a,epsilon,N,fileName):
    #plt.clf()

    trainingLoss, trainingGradient, w, SSE = gradientDescent(y, w, x, l, a, epsilon, N)
    x, y = generateVariables(testingData)
    testingLoss, testingGradient, w, SSE = gradientDescentTesting(y, w, x, 0, N)

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
    #plt.ylim((0,60000))
    #plt.xlim((-5,100))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(pp, format='pdf',bbox_inches = 'tight')
    pp.close()
    #plt.clf()

    pp = PdfPages('trainingGradient.pdf')
    plt.figure(1)
    #plt.ylim((0,10000))
    #plt.xlim((-5,100))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(pp, format='pdf',bbox_inches = 'tight')
    pp.close()

    pp = PdfPages('testingGradient.pdf')
    plt.figure(2)
    #plt.ylim((0,1000))
    #plt.xlim((-5,100))
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

def problem2TrainandTest(y,w,x,l,a,epsilon,N,testingSSElist, trainingSSElist):
    #plt.clf()
    
    trainingLoss, trainingGradient, w, trainingSSE = gradientDescent(y, w, x, l, a, epsilon, N)
    x, y = generateVariables(testingData)
    testingLoss, testingGradient, w, testingSSE = gradientDescentTesting(y, w, x, l, N)
    
    testingSSElist.append(testingSSE)
    trainingSSElist.append(trainingSSE)

    plt.figure(0)
    plt.plot(trainingLoss, label='l='+str(l))
    
    plt.figure(1)
    plt.plot(trainingGradient, label='l='+str(l))

    plt.figure(2)
    plt.plot(testingGradient, label='l='+str(l))
    
    plt.figure(3)
    plt.plot(testingLoss, label='l='+str(l))

    return trainingSSElist,testingSSElist

def problem2(y, w, x, l, a, epsilon, N):
    trainingSSElist = list()
    testingSSElist = list()

    trainingSSElist, testingSSElist = problem2TrainandTest(y,w,x,0,0.001,epsilon,N, testingSSElist, trainingSSElist)
    trainingSSElist, testingSSElist = problem2TrainandTest(y,w,x,0.000001,0.001,epsilon,N, testingSSElist, trainingSSElist)
    trainingSSElist, testingSSElist = problem2TrainandTest(y,w,x,0.00001,0.001,epsilon,N, testingSSElist, trainingSSElist)
    trainingSSElist, testingSSElist = problem2TrainandTest(y,w,x,0.0001,0.001,epsilon,N, testingSSElist, trainingSSElist)
    trainingSSElist, testingSSElist = problem2TrainandTest(y,w,x,0.001,0.001,epsilon,N, testingSSElist, trainingSSElist)
    trainingSSElist, testingSSElist = problem2TrainandTest(y,w,x,0.01,0.001,epsilon,N, testingSSElist, trainingSSElist)
    trainingSSElist, testingSSElist = problem2TrainandTest(y,w,x,0.1,0.001,epsilon,N, testingSSElist, trainingSSElist)
    trainingSSElist, testingSSElist = problem2TrainandTest(y,w,x,1,0.001,epsilon,N, testingSSElist, trainingSSElist)
    trainingSSElist, testingSSElist = problem2TrainandTest(y,w,x,10,0.001,epsilon,N, testingSSElist, trainingSSElist)
    trainingSSElist, testingSSElist = problem2TrainandTest(y,w,x,100,0.001,epsilon,N, testingSSElist, trainingSSElist)

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

    pp = PdfPages('SSE.pdf')
    plt.figure(4)
    print testingSSElist
    lambList = [0,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100]
    plt.plot(lambList, testingSSElist, label='testing')
    plt.plot(lambList, trainingSSElist, label='training')
    plt.xscale('log')
    #plt.ylim((0,1000))
    #plt.xlim((-1,1))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(pp, format='pdf',bbox_inches = 'tight')
    pp.close()
	

def problem3(y, x, a):
    train_x = []
    train_y = []

    test_x = []
    test_y = []

    test_x = np.split(x, 10)
    test_y = np.split(y, 10)
    print test_x[9].shape
    for i in range(10):
        temp_list = []
        for j in range(10):
            if (i != j):
                temp_list.append(test_x[j])

        all_train_arrs = tuple(temp_list)
        train_x.append(np.vstack(all_train_arrs))

    lambda_vals = [0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 5.0, 10.0]
    min_test_loss = -1



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
    #problem1(y, w, x, l, a, epsilon, N)
    #problem2(y, w, x, l, a, epsilon, N)
    #problem1TrainandTest(y,w,x,l,0.001,epsilon,N, 'a001')
    #plt.show()
    #problem2()
    problem3(y, x, a)

# Giving Python the main it deserves
if __name__ == "__main__": 
	main()
