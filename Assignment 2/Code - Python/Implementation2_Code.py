from __future__ import division
import csv
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from sys import argv
from copy import copy, deepcopy
from numpy.linalg import inv
import random
import math

# Not sure what script is for...
#script = 'data/'
newsgrouplabels = 'data/newsgrouplabels.txt'
vocabulary = 'data/vocabulary.txt'
testLabels = 'data/test.label'
testData = 'data/test.data'
trainLabels = 'data/train.label'
trainData = 'data/train.data'

#########################################################################################################################
############ 									Take in our data files                           ########################
############ Returns: Group_Labels array, Vocab array, Data_Labels 1-D array, Data_Data 2-D array########################
def readData(newsgrouplabels, vocabulary, data_labels, data_data):
	newsGroup = list()
	vocab = list()

	for line in open(newsgrouplabels,'r').readlines():
		newsGroup.append(line.strip())
	
	for line in open(vocabulary,'r').readlines():
		vocab.append(line.strip())
	
	dataLabels = np.genfromtxt(data_labels,delimiter='')
	
	actualData = np.genfromtxt(data_data,delimiter='')

	return newsGroup, vocab, dataLabels, actualData
	
###############################################################################################################################################################
#######Create a 2-D array that shows # of word occurrences per class, also 2-D array that counts number of documents containing a certain word#################
def Class_Word_Matrix(Group_Labels, Vocab, Data_Labels, Data_Data):
	Class_WC = np.zeros((len(Group_Labels), len(Vocab))) #Number of words in vocab by number of classes
	Document_Word_Occur = np.zeros((len(Group_Labels), len(Vocab))) #Number of words in vocab by number of classes
	Doc_Num_Each_Class = [0]*20 #Count how many documents per class exist in train.label
	count = 0
	current_word = 0
	word_count = 0
	Total_Docs_Per_Class = [0]*20
	Py = [0]*20
	
	#loop through entire data label array#
	for x in xrange(0, len(Data_Labels)):
		current_class = int(Data_Labels[x]) - 1 #Capture class label --1 through 20-- represented by --0 through 19--
		Total_Docs_Per_Class[current_class] += 1 #As we iterate through train.label, iterate how many documents exist in the current class
		#Doc_Num_Each_Class[current_class] = Total_Docs_Per_Class #Update the 1-D array keeping track of #documents for each class
		
		#if x <= len(Data_Labels) and Data_Labels[x + 1] != current_class + 1: #Ensure we don't look at a non-existent array index AND reset our document counter per class when the next index sees a new class coming up
		#	Total_Docs_Per_Class = 0

		#For each particular document, loop through every word it contains#
		while Data_Data[count][0] == x + 1: 
			current_word = Data_Data[count][1] - 1 #Keep track of the current word we're on, note the -1 is rearranging the word to match correct index.... Word 1 in vocabulary matches to index 0 and so on.
			word_count = Data_Data[count][2] #How many times does certain word for current document occur
			Class_WC[current_class][current_word] = Class_WC[current_class][current_word] + word_count #Keep track of the number of a particular word occurring in a certain class
			
			if word_count > 0:
				Document_Word_Occur[current_class][current_word] = Document_Word_Occur[current_class][current_word] + 1 #Other array that keeps track of number of documents that contain a certain word in a class
			
			if count < len(Data_Data) - 1: #Keep incrementing our count value to keep going through our data file
				count = count + 1
			else:
				break
	
	
	Total_Docs_Per_Class_array = np.asarray(Total_Docs_Per_Class)
	Total_Docs_Per_Class_array = Total_Docs_Per_Class_array.astype(float)
	Py = Total_Docs_Per_Class_array/sum(Total_Docs_Per_Class_array)
	#print Doc_Num_Each_Class  verify values returned
	#print Class_WC[19][12101] verify values returned
	#print Document_Word_Occur[19][12101] verify values returned
	return Class_WC, Document_Word_Occur, Total_Docs_Per_Class, Py

def wordOccuredMatrix(numClasses, numWords, data):
	wordOccured = np.zeros((7505, numWords))
	for row in data:
		docClass = row[0]
		word = row[1]
		count = row[2]
		if count > 0:
			wordOccured[docClass-1][word-1] = 1

	return wordOccured
	

def BernouliTrain(Document_Word_Occur, Total_Docs_Per_Class, vocab, alpha, beta):
	py = [0] *20
	px_y = np.copy(Document_Word_Occur)
	top = alpha - 1
	bottom = alpha + beta - 2
	doesWordOccur = np.copy(Document_Word_Occur)

	for docClass in xrange(0,20):
		py[docClass] = (Total_Docs_Per_Class[docClass] + top)/(sum(Total_Docs_Per_Class) + bottom)

	for docClass in xrange(0,20):
		for word in xrange(0, len(vocab)):
			if Document_Word_Occur[docClass][word] > 0:
				doesWordOccur[docClass][word] = 1

			px_y[docClass][word] = (1 + Document_Word_Occur[docClass][word])/ (len(vocab) + Total_Docs_Per_Class[docClass])
			#print px_y[docClass][word]

	return py, px_y


def Bernouli_Laplace(Document_Word_Occur, Total_Docs_Per_Class, vocab, alpha, beta):	
	Pi_y = copy.deepcopy(Document_Word_Occur)
	Top = alpha - 1
	Bottom = alpha + beta - 2
	Px_y = [0]*20

	for x in xrange(0,20):
		Pi_y[x] = (Pi_y[x] + Top)/(Total_Docs_Per_Class[x] + Bottom)

	doesWordOccur = np.zeros(20, len(vocab))
	for y in xrange(0,20):
		for z in xrange(0, len(vocab)):
			if Document_Word_Occur[y][z] > 0:
				doesWordOccur[y][z] = 1
		
			Px_y[y] += math.log((Pi_y[y][z]**(doesWordOccur[y][z]))*((1 - Pi_y[y][z])**(1 - doesWordOccur[y][z])), 2)
		#print Px_y[y]

		
def BernouliTest(wordOccured, py, px_y, numClasses, numWords):
	product = 0
	docClassPrediction = -1
	docClassProbability = float("-inf")
	for docClass in xrange(0,numClasses):
		#product = py[docClass]
		product = 0
		for word in xrange(0,numWords):
			if (px_y[docClass][word]*(wordOccured[word])) + ((1 - px_y[docClass][word]) * (1-wordOccured[word])) == 0:
				print px_y[docClass][word],(wordOccured[word]), (1 - px_y[docClass][word]), (wordOccured[word])
			#print product
			#product *= (px_y[docClass][word]*(wordOccured[word])) + ((1 - px_y[docClass][word])*(1-wordOccured[word]))
			product += math.log((px_y[docClass][word]**(wordOccured[word])) * ((1 - px_y[docClass][word])**(1-wordOccured[word])),2)
		#print product
		if product > docClassProbability:
			#print product
			docClassPrediction = docClass
			docClassProbability = product

	return docClassPrediction
				
def Multinomial_Laplace(Class_WC, vocab, alpha):
	Pi_y2 = Class_WC
	ClassWords = [0]*20
	for x in xrange(0,20):
		ClassWords[x] = sum(Class_WC[x])
		Pi_y2[x] = (Pi_y2[x] + alpha)/(ClassWords[x] + alpha*len(vocab))
		
	
def problem1(py, px_y, numClasses, numWords):
	print "Reading in testing data..."
	testingLabels = np.genfromtxt(testLabels,delimiter='')
	testingData = np.genfromtxt(testData,delimiter='')
	print "Testing Data Read Complete"
	print "Calculating word occured matrix..."
	wordOccured = wordOccuredMatrix(numClasses, numWords, testingData)
	print "Word Occured Matrix Complete"
	totalCorrect = 0
	totalDocs = 0
	for docNum, doc in enumerate(wordOccured):
		totalDocs += 1 
		print "Predicting Document %d..." % (docNum)
		predictedClass = BernouliTest(doc, py, px_y, numClasses, numWords)
		print predictedClass, testingLabels[docNum]
		if testingLabels[docNum]-1 == predictedClass:
			print "Correct"
		else:
			print "Wrong!!"


def problem2():
	pass

def problem3():
	pass
	
def main():
	alpha = beta = 2
	print "Starting..."
	print "Reading in data..."
	newsGroup, vocab, dataLabels, actualData = readData(newsgrouplabels, vocabulary, trainLabels, trainData)
	print "Data Read Complete"
	print "Counting data..."
	Class_WC, Document_Word_Occur, Total_Docs_Per_Class, Py = Class_Word_Matrix(newsGroup, vocab, dataLabels, actualData)
	print "Counting Data Complete"
	print "Calculating probabilities..."
	py, px_y = BernouliTrain(Document_Word_Occur, Total_Docs_Per_Class, vocab, alpha, beta)
	print "Probabilities Complete"
	#Bernouli_Laplace(Document_Word_Occur, Total_Docs_Per_Class, vocab, alpha, beta)
	#Multinomial_Laplace(Class_WC, vocab, alpha)
	
	problem1(py, px_y, len(newsGroup), len(vocab))
	#problem2()
	#problem3()


# Giving Python the main it deserves
if __name__ == "__main__": 
	main()	
