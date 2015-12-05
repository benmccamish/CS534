from __future__ import division
import csv
import math
import random
import copy as cp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#Graphs the Eigen Values
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

#Graphs the cummulative varience of eigen values
def plotLineGraph(cumLineData, eigValues, percentage):
	plt.axhline(y=percentage, xmin=0, xmax=1, hold=None)
	plt.plot(cumLineData, color='r',)
	plt.ylabel('Cumulative Distribution')
	plt.xlabel('Principle Component')
	pp = PdfPages('CDF_Varience.pdf')
	plt.savefig(pp, format='pdf')
	pp.close()

#Gets the top eigen value and vector pairs that have a cummulative variance above that which is specified.
def CalculatePercentageEigen(percentage, eigValVect):
	eigValues = [float(i[0]) for i in eigValVect]
	totalEigValues = sum(eigValues)
	varience = []
	for val in eigValues:
		varience.append(float(val)/totalEigValues)

	cumVarience = np.cumsum(varience)
	plotLineGraph(cumVarience, eigValues, percentage)

	relevantEigVals = 0

	for i in range(len(cumVarience)):
		if(cumVarience[i] >= percentage):
			relevantEigVals = i
			break

	plotBarGraph(cumVarience, eigValues, percentage, relevantEigVals)
	
	return eigValVect[:relevantEigVals]

#Sorts eigen values with their vectors in decreasing order
def EigValVectSorted(eigValues, eigVectors):
	eigValVect = []
	for i in range(len(eigValues)):
		eigValVect.append((abs(eigValues[i]), eigVectors[:,i]))

	eigValVect.sort()
	eigValVect.reverse()
	return eigValVect

#Gets the top K eigen vectors and puts them into a 256 x K matrix used to reduced the dimensions of the
def TopKEiganVectMatrix(k, eigValVect):
	eigVectReduction = (eigValVect[0][1])[None].T
	for i in xrange(1,k):
		eigVectReduction = np.hstack((eigVectReduction, (eigValVect[i][1])[None].T))

	return eigVectReduction

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

#Calculates and returns the centroid of the given cluster
def Cluster_Centroid(Cluster):
	Centroid = [0]*len(Cluster[0])
	
	Cluster_componenets = map(list, zip(*Cluster))

	for x in range(0, len(Cluster_componenets)):
		Centroid[x] = float(sum(Cluster_componenets[x]))/len(Cluster_componenets[x])

	return Centroid


#Takes in data and places each data point in the cluster that yields the lowest SSE.  Example labels are kept track of in each cluster by being placed in the respective label1 or label2 lists 	
def reCluster(centroid1, centroid2, data, labels):
	cluster1 = []
	cluster2 = []
	labels1 = []
	labels2 = []
	totalError = 0
	for x in range(0, len(data)):
		sse1 = 0
		sse2 = 0
		for y in range(0, len(data[0])):
			sse1 += (data[x][y] - centroid1[y])**2
			sse2 += (data[x][y] - centroid2[y])**2
		if sse1 > sse2:
			cluster2.append(data[x])
			labels2.append(labels[x])
			totalError += sse2
		else:
			cluster1.append(data[x])
			labels1.append(labels[x])
			totalError += sse1
		
	return cluster1, cluster2, labels1, labels2, totalError
			
	
#Two clusters are randomly initialized, each with a single data point.  Clusters change based on minimizing SSE and updating cluster centroids. 
#The two optimal clusters are returned along with corresponding labels, total error from points to their cluster, and the purity of each cluster. 
#NOTE: This function only runs kmeans once.  kmeansTenTimes is created to call kmeans ten times.
def kmeans(k, data, labels):
	cluster1 = []
	cluster2 = []
	labels1 = []
	labels2 = []
	sevenPurity = 0
	ninePurity = 0
	Initial_points = random.sample(data, 2)
	#print Initial_points[0]
	
	cluster1.append(Initial_points[0])
	cluster2.append(Initial_points[1])
	Temp = 0
	
	centroid1 = Cluster_Centroid(cluster1)

	while Temp != len(cluster1):
		#print len(cluster1)
	
		centroid1 = Cluster_Centroid(cluster1)
		centroid2 = Cluster_Centroid(cluster2)
		Temp = len(cluster1)
		
		cluster1, cluster2, labels1, labels2, totalError = reCluster(centroid1, centroid2, data, labels)
		#print totalError
		
	
	if labels1.count(9) > labels1.count(7):
		ninePurity = labels1.count(9)/len(labels1)
		sevenPurity = labels2.count(7)/len(labels2)
		#print 'The accuracy for class 9 was %f and the accuracy for class 7 was %f' %(ninePurity, sevenPurity) 
		#print 'First cluster is nine with %d out of %d correct' %(labels1.count(9), len(labels1))
		#print 'Second cluster is seven with %d out of %d correct' %(labels2.count(7), len(labels2))
	else:
		sevenPurity = labels1.count(7)/len(labels1)
		ninePurity = labels2.count(9)/len(labels2)		
		#print 'The accuracy for class 9 was %f and the accuracy for class 7 was %f' %(ninePurity, sevenPurity)
		#print 'First cluster is seven with %d out of %d correct' %(labels1.count(7), len(labels1))
	
	
	
	
	return cluster1, cluster2, labels1, labels2, totalError, ninePurity, sevenPurity

#Function that just calls kmeans 10 times.  It finds the best two clusters, 	
def kmeansTenTimes(k, data, labels):
	lowestTotalError = float("inf")
	sevenPurity = 0
	ninePurity = 0
	for x in range(0, 10):
		cluster1, cluster2, labels1, labels2, totalError, unused1, unused2 = kmeans(k, data, labels)
		print 'Kmeans run resulted in a total SSE (incluses SSE of all points) of %f' %totalError
		if totalError < lowestTotalError:
			bestCluster1 = cluster1
			bestCluster2 = cluster2
			bestCluster1Labels = labels1
			bestCluster2Labels = labels2
			lowestTotalError = totalError

	
	if bestCluster1Labels.count(9) > bestCluster1Labels.count(7):
		ninePurity = bestCluster1Labels.count(9)/len(bestCluster1Labels)
		sevenPurity = bestCluster2Labels.count(7)/len(bestCluster2Labels)

	else:
		sevenPurity = bestCluster1Labels.count(7)/len(bestCluster1Labels)
		ninePurity = bestCluster2Labels.count(9)/len(bestCluster2Labels)
		


	return bestCluster1, bestCluster2, bestCluster1Labels, bestCluster2Labels, lowestTotalError, ninePurity, sevenPurity
	
	
def Problem1(k, data, labels):
	#cluster1, cluster2, labels1, labels2, totalError, ninePurity, sevenPurity = kmeans(k, data, labels)
	bestCluster1, bestCluster2, bestCluster1Labels, bestCluster2Labels, lowestTotalError, ninePurity, sevenPurity = kmeansTenTimes(k, data, labels)
	print 'The accuracy for class 9 was %f and the accuracy for class 7 was %f' %(ninePurity, sevenPurity)
	print 'lowest total error found was %f' %lowestTotalError		

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
	print "Getting top %d eigen vectors" % (reducedToDim)
	reduction = TopKEiganVectMatrix(reducedToDim, eigValVect)
	print "Reducing data..."
	reducedData = data.dot(reduction)
	print "Data Reduced"
	print "Calculating KMeans with reduced data"
	Problem1(k, reducedData, labels)
	print "Problem 3 Done\n\n"

def Problem4(data, labels):

	class_7 = []
	class_9 = []

	for i in range(len(labels)):
		if (labels[i] == 7):
			class_7.append(data[i])
		else:
			class_9.append(data[i])

	mean_7 = np.mean(class_7, axis=0)
	mean_9 = np.mean(class_9, axis=0)
	print(mean_7)

	S = np.matrix(np.zeros((256, 256)))

	#print(class_7[0].shape)
	for i in range(len(class_7)):
		my_mat = np.matrix((class_7[i] - mean_7))
		S = S + (my_mat.T * my_mat)

	for i in range(len(class_9)):
		my_mat = np.matrix((class_9[i] - mean_9))
		S = S + (my_mat.T * my_mat)

	print( (np.multiply((class_9[i] - mean_9), np.transpose((class_9[i] - mean_9)))).size)
	w =  (mean_7 - mean_9) * np.linalg.inv(S)
	
	print(w)
	print(w.size)
	print(type(w))

	projected_data = []

	for i in range(len(labels)):
		projected_pt = np.matrix(data[i]) * w.T
		print(projected_pt)
		
		projected_data.append(projected_pt)

	cluster1, cluster2, labels1, labels2, totalError, ninePurity, sevenPurity = kmeans(2, projected_data, labels)

	print(ninePurity)
	print(sevenPurity)	

def main():
	dataFile = 'Data.csv'
	labelsFile = 'Labels.csv'
	k = 2
	
	data, labels = Import_Data(dataFile, labelsFile)
	
	#Problem1(k, data, labels)
	
	#Problem2(data, labels, 0.9)

	Problem3(42, k, data, labels)

	#Problem4(data, labels)


if __name__ == "__main__":
	main()
