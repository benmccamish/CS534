from __future__ import division
import csv
import numpy as np
from sys import argv
from copy import copy, deepcopy
from numpy.linalg import inv
import random
import copy
import math
import operator

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

def zipfsLaw(vocab, actualData):
	dictOfWords = {}
	listOfWords = []
	wordIndex = {}
	for word in vocab:
		listOfWords.append(word)
		dictOfWords[word.strip()] = 0

	for row in actualData:
		dictOfWords[listOfWords[int(row[1])]] += int(row[2])

	return listOfWords, sorted(dictOfWords.items(), key=operator.itemgetter(1), reverse = True)

def removeWords(listOfWords, actualData):
	for word in listOfWords:
		actualData = actualData[actualData[:,1] != word+1]
	return actualData

def removeTopX(actualData, zipfsWords, x, listOfWords):
	topXWords = []
	wordIndexes = list()
	print "Top %d Words Are:" % (x)
	for i in xrange(0, x):
		print i+1, zipfsWords[i]
		topXWords.append(zipfsWords[i][0])

	for word in topXWords:
		wordIndexes.append(listOfWords.index(word)+1)
		actualData = actualData[actualData[:,1] != listOfWords.index(word)+1]
		
	return wordIndexes, actualData

def performZipfsFilter(vocab, actualData, x):
	listOfWords, zipfsWords = zipfsLaw(vocab, actualData)
	return removeTopX(actualData, zipfsWords, x, listOfWords)

def wordOccuredMatrix(numClasses, numWords, data):
	wordOccured = np.zeros((7505, numWords))
	wordOccur_count = np.zeros((7505, numWords))
	wordList = [ [] for i in range(7505)]
	print len(wordList)
	for row in data:
		docClass = row[0]
		word = row[1]
		count = row[2]
		if count > 0:
			wordOccured[docClass-1][word-1] = 1
			wordOccur_count[docClass-1][word-1] = count
		wordList[int(docClass) -1].append(word);

	return wordOccured, wordList, wordOccur_count
	
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

			px_y[docClass][word] = (top + Document_Word_Occur[docClass][word])/ (bottom + Total_Docs_Per_Class[docClass])
			#print px_y[docClass][word]

	return py, px_y
		
def BernouliTest(wordOccured, total_log, numClasses, biases, wordList, Py):
	product = 0
	docClassPrediction = -1
	docClassProbability = float("-inf")
	
	for docClass in xrange(0,numClasses):
		product = biases[docClass] + Py[docClass]
		for word in wordList:
			product += total_log[docClass][1][word - 1] - total_log[docClass][0][word - 1]

		if product > docClassProbability:
			docClassPrediction = docClass
			docClassProbability = product

	return docClassPrediction
				
def Multinomial_Train(Class_WC, vocab, alpha):
	Pi_y2 = Class_WC
	ClassWords = [0]*20
	for x in xrange(0,20):
		ClassWords[x] = sum(Class_WC[x])
		Pi_y2[x] = (Pi_y2[x] + alpha)/(ClassWords[x] + alpha*len(vocab))
		
	return Pi_y2

def Multinomial_Test(doc, log_total, numClasses, wordList, wordOccur_count, numWords, Py):
	product = 0
	counter = -1
	docClassPrediction = -1
	docClassProbability = float("-inf")
	new = [wordOccur_count[i-1] for i in wordList]
	
	for docClass in xrange(0,numClasses):
		
		product = Py[docClass]
		counter = 0
		for x in wordList:
			product += log_total[docClass][0][x - 1]*new[counter]
			counter += 1
		if product > docClassProbability:
			docClassPrediction = docClass
			docClassProbability = product

	return docClassPrediction		
		
def problem1(py, px_y, numClasses, numWords, Py, wordIndexes):
	print "Reading in testing data..."
	testingLabels = np.genfromtxt(testLabels,delimiter='')
	testingData = np.genfromtxt(testData,delimiter='')
	testingData = removeWords(wordIndexes, testingData)
	print "Testing Data Read Complete"
	print "Calculating word occured matrix..."
	wordOccured, wordList, wordOccur_count = wordOccuredMatrix(numClasses, numWords, testingData)
	print "Word Occured Matrix Complete"
	totalCorrect = 0
	totalDocs = 0
	right = 0
	confusion_matrix = np.zeros((20, 20))
	
	total_log = []
	biases = []
	for c in range(len(px_y)):
		log_positive = np.log(px_y[c])
		log_negative = np.log(1 - px_y[c])
		biases.append(np.sum(log_negative))
		class_log = [log_negative, log_positive]
		total_log.append(class_log)

	for docNum, doc in enumerate(wordOccured):
		totalDocs += 1 
		print "Predicting Document %d..." % (docNum)
		
		predictedClass = BernouliTest(doc, total_log, numClasses, biases, wordList[docNum], Py)
		print predictedClass + 1, testingLabels[docNum]
		if testingLabels[docNum]-1 == predictedClass:
			print "Correct"
			right += 1
		else:
			print "Wrong!!"
		confusion_matrix[testingLabels[docNum]-1][predictedClass] += 1
	print right		
	np.savetxt('Bernoulli_Confusion.csv', confusion_matrix, delimiter=',')

def problem2(Py, Pi_y2, numClasses, numWords, wordIndexes):
	print "Reading in testing data..."
	testingLabels = np.genfromtxt(testLabels,delimiter='')
	testingData = np.genfromtxt(testData,delimiter='')
	testingData = removeWords(wordIndexes, testingData)
	print "Testing Data Read Complete"
	print "Calculating word occured matrix..."
	wordOccured, wordList, wordOccur_count = wordOccuredMatrix(numClasses, numWords, testingData)
	print "Word Occured Matrix Complete"
	totalCorrect = 0
	totalDocs = 0
	right = 0
	confusion_matrix2 = np.zeros((20, 20))

        
	log_total = []
	print Pi_y2
	
	for a in range(len(Pi_y2)):
		logarithmic = np.log(Pi_y2[a])
		class_log = [logarithmic]
		log_total.append(class_log)

	for docNum, doc in enumerate(wordOccured):
		totalDocs += 1 
		print "Predicting Document %d..." % (docNum)

		predictedClass = Multinomial_Test(doc, log_total, numClasses, wordList[docNum], wordOccur_count[docNum], numWords, Py)
		print predictedClass + 1, testingLabels[docNum]
		if testingLabels[docNum]-1 == predictedClass:
			print "Correct"
			right += 1
          
                
		else:
			print "Wrong!!"
		confusion_matrix2[testingLabels[docNum]-1][predictedClass] += 1

	print right	
	np.savetxt('Multinomial_Confusion.csv', confusion_matrix2, delimiter=',')	

def problem3():
	pass
	
def main():
	alpha = beta = 2
	alpha2 = 1
	print "Starting..."
	print "Reading in data..."
	newsGroup, vocab, dataLabels, actualData = readData(newsgrouplabels, vocabulary, trainLabels, trainData)
	print "Data Read Complete"

	#Comment this out if you want to run without zipfs
	print "Performing Zipfs Law"
	wordIndexes, actualData = performZipfsFilter(vocab, actualData, 300)
	print "Zipfs Law Complete"
	
	print "Counting data..."
	Class_WC, Document_Word_Occur, Total_Docs_Per_Class, Py = Class_Word_Matrix(newsGroup, vocab, dataLabels, actualData)
	print "Counting Data Complete"
	print "Calculating probabilities..."
	py, px_y = BernouliTrain(Document_Word_Occur, Total_Docs_Per_Class, vocab, alpha, beta)
	print "Probabilities Complete"
	Pi_y2 = Multinomial_Train(Class_WC, vocab, alpha2)
	
	problem1(py, px_y, len(newsGroup), len(vocab), Py, wordIndexes)
	#problem2(Py, Pi_y2, len(newsGroup), len(vocab), wordIndexes)
	#problem3()


# Giving Python the main it deserves
if __name__ == "__main__": 
	main()	
