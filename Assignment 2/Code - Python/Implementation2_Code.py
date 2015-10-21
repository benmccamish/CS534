import csv
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from sys import argv
from copy import copy, deepcopy
from numpy.linalg import inv
import random

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
	Class_WC = np.zeros((len(newsgrouplabels), len(Vocab))) #Number of words in vocab by number of classes
	Document_Word_Occur = np.zeros((len(Group_Labels), len(Vocab))) #Number of words in vocab by number of classes
	Doc_Num_Each_Class = [0]*20 #Count how many documents per class exist in train.label
	count = 0
	current_word = 0
	word_count = 0
	Total_Docs_Per_Class = [0]*20
	
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

	#print Doc_Num_Each_Class  verify values returned
	#print Class_WC[19][12101] verify values returned
	#print Document_Word_Occur[19][12101] verify values returned
	return Class_WC, Document_Word_Occur, Doc_Num_Each_Class


def Bernouli_Laplace(Document_Word_Occur):	
	print 'nothing'
	
def problem1():
	pass

def problem2():
	pass

def problem3():
	pass
	
def main():
	newsGroup, vocab, dataLabels, actualData = readData(newsgrouplabels, vocabulary, data_labels, data_data)
	Class_WC, Document_Word_Occur, Doc_Num_Each_Class = Class_Word_Matrix(Group_Labels, Vocab, Data_Labels, Data_Data)


	#problem1()
	#problem2()
	#problem3()


# Giving Python the main it deserves
if __name__ == "__main__": 
	main()	
