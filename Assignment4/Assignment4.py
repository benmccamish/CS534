from __future__ import division
import csv
import math
import random
import copy as cp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def plotBarGraph(cumLineData, eigValues, percentage, relevantEigVals):
	threshold = percentage
	values = np.array(eigValues[:relevantEigVals])
	# split it up
	above_threshold = np.maximum(values - threshold, 0)
	below_threshold = np.minimum(values, threshold)

	x = range(len(values))

	# and plot it
	fig, ax = plt.subplots()
	ax.bar(x, below_threshold, 0.35, color="g")
	ax.bar(x, above_threshold, 0.35, color="r",bottom=below_threshold)

	ax.plot([0., relevantEigVals], [threshold, threshold], "k--")

	fig.savefig("BarGraph.pdf")


def plotLineGraph(cumLineData, eigValues, percentage):
	plt.axhline(y=percentage, xmin=0, xmax=1, hold=None)
	plt.plot(cumLineData, color='r',)
	plt.ylabel('Cumulative Distribution')
	plt.xlabel('Principle Component')
	pp = PdfPages('CDF_Varience.pdf')
	plt.savefig(pp, format='pdf')
	pp.close()


def CalculatePercentageEigen(percentage, eigValVect):
	eigValues = [float(i[0]) for i in eigValVect]
	totalEigValues = sum(eigValues)
	varience = []
	for val in eigValues:
		varience.append(float(val)/totalEigValues)

	cumVarience = np.cumsum(varience)
	plotLineGraph(cumVarience, eigValues, percentage)
	#plotBarGraph(cumVarience, eigValues, percentage)

	relevantEigVals = 0

	for i in range(len(cumVarience)):
		if(cumVarience[i] >= percentage):
			relevantEigVals = i
			break

	plotBarGraph(cumVarience, eigValues, percentage, relevantEigVals)
	
	return eigValVect[:relevantEigVals]

def EigValVectSorted(eigValues, eigVectors):
	eigValVect = []
	for i in range(len(eigValues)):
		eigValVect.append((abs(eigValues[i]), eigVectors[:,i]))

	eigValVect.sort()
	eigValVect.reverse()
	return eigValVect

def TopKEiganVectMatrix(k, eigValVect):
	eigVectReduction
	for i in xrange(1,k):
		print i



	#return eigValVect[:k]

#Gets the Covarance or Scatter matrix of the data, both produce equiv i matrix
def CovarianceMatrix(data):
	#Get the average x
	avg = np.mean(data, axis=0)
	covMatrix = 0
	for i in range(data.shape[0]):
		covMatrix += (data[i] - avg)*((data[i] - avg)[None].T)

	covMatrix = covMatrix/(float(1)/data.shape[0])
	return covMatrix

def Import_Data(Data, Labels):
	Data_matrix = np.genfromtxt(Data,delimiter=',')
	Labels_vector = np.genfromtxt(Labels,delimiter=',')
	
	return Data_matrix, Labels_vector

def Cluster_Centroid(Cluster):
	Centroid = [0]*len(Cluster[0])
	
	Cluster_componenets = map(list, zip(*Cluster))

	for x in range(0, len(Cluster_componenets)):
		Centroid[x] = float(sum(Cluster_componenets[x]))/len(Cluster_componenets[x])

	return Centroid
	
def Kmeans_Helper(Centroid_1, Centroid_2, Data_matrix, Labels_1, Labels_2, Labels_vector):
	Cluster_1 = []
	Cluster_2 = []
	Labels_1 = []
	Labels_2 = []
	for x in range(0, len(Data_matrix)):
		SSE1 = 0
		SSE2 = 0
		for y in range(0, len(Data_matrix[0])):
			SSE1 += (Data_matrix[x][y] - Centroid_1[y])**2
			SSE2 += (Data_matrix[x][y] - Centroid_2[y])**2
		if SSE1 > SSE2:
			Cluster_2.append(Data_matrix[x])
			Labels_2.append(Labels_vector[x])
		else:
			Cluster_1.append(Data_matrix[x])
			Labels_1.append(Labels_vector[x])
			
	return Cluster_1, Cluster_2, Labels_1, Labels_2
			
	
	
def Kmeans(k, Data_matrix, Labels_vector):
	Cluster_1 = []
	Cluster_2 = []
	Labels_1 = []
	Labels_2 = []
	Initial_points = random.sample(Data_matrix, 2)
	
	Cluster_1.append(Initial_points[0])
	Cluster_2.append(Initial_points[1])
	Temp = 0
	
	Centroid_1 = Cluster_Centroid(Cluster_1)

	while Temp != len(Cluster_1):
	
		Centroid_1 = Cluster_Centroid(Cluster_1)
		Centroid_2 = Cluster_Centroid(Cluster_2)
		Temp = len(Cluster_1)
		
		Cluster_1, Cluster_2, Labels_1, Labels_2 = Kmeans_Helper(Centroid_1, Centroid_2, Data_matrix, Labels_1, Labels_2, Labels_vector)

	if Labels_1.count(9) > Labels_1.count(7):
		print 'First cluster is nine with %d out of %d correct' %(Labels_1.count(9), len(Labels_1))
		print 'Second cluster is seven with %d out of %d correct' %(Labels_2.count(7), len(Labels_2))
	else:
		print 'First cluster is seven with %d out of %d correct' %(Labels_1.count(7), len(Labels_1))
		print 'Second cluster is nine with %d out of %d correct' %(Labels_2.count(9), len(Labels_2))		
	

def Problem1(k, data, labels):
	Kmeans(k, data, labels)

def Problem2(data, labels, percentage):
	print "\n\nStarting Problem 2"
	print "Getting Covariance Matrix..."
	covMatrix = CovarianceMatrix(data)
	print "Covariance Matrix Done"
	print "Getting Eigen Values and Vectors..."
	eigValues, eigVectors = np.linalg.eig(covMatrix)
	print "Eigen Values and Vectors Done"
	print "Sorting..."
	eigValVect = EigValVectSorted(eigValues, eigVectors)
	print 'Done'
	print "Graphing and selecting eigen values that have cummulative varience greater than %f" % (percentage)
	eigValVect = CalculatePercentageEigen(percentage, eigValVect)
	print "Problem 2 Done\n\n"

def Problem3(reducedToDim, k, data, labels):
	print "\n\nStarting Problem 3"
	print "Getting Covariance Matrix..."
	covMatrix = CovarianceMatrix(data)
	print "Covariance Matrix Done"
	print "Getting Eigen Values and Vectors..."
	eigValues, eigVectors = np.linalg.eig(covMatrix)
	print "Eigen Values and Vectors Done"
	print "Sorting..."
	eigValVect = EigValVectSorted(eigValues, eigVectors)
	print 'Done'
	TopKEiganVectMatrix(reducedToDim, eigValVect)

def main():
	dataFile = 'Data.csv'
	labelsFile = 'Labels.csv'
	k = 2
	
	data, labels = Import_Data(dataFile, labelsFile)
	
	#Problem1(k, data, labels)
	
	#Problem2(data, labels, 0.9)

	Problem3(10, k, data, labels)


if __name__ == "__main__":
	main()
