from __future__ import division
import csv
import math
import random
import copy as cp
import numpy as np



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

		print 'Second cluster is seven with %d out of %d correct' %(Labels_2.count(7), len(Labels_2)
	else:
		print 'First cluster is seven with %d out of %d correct' %(Labels_1.count(7), len(Labels_1))
	
		print 'Second cluster is nine with %d out of %d correct' %(Labels_2.count(9), len(Labels_2))		
	
	

def main():
	Data = 'Data.csv'
	Labels = 'Labels.csv'
	k = 2
	
	Data_matrix, Labels_vector = Import_Data(Data, Labels)
	Kmeans(k, Data_matrix, Labels_vector)
	
if __name__ == "__main__":
	main()
