from __future__ import division
import csv
import math
import random
import copy as cp
import numpy as np
import matplotlib.pyplot as plt



def read_csv_data(train_name, test_name):

	Train_X = []
	Train_Y = []
	
	with open(train_name, 'r') as csvfile:
		my_reader = csv.reader(csvfile)
		for row in my_reader:
			row = [float(i) for i in row]
			Train_Y.append(row[0])
			Train_X.append(row[1:])
	
	Test_X = []
	Test_Y = []

	with open(test_name, 'r') as csvfile:
		my_reader = csv.reader(csvfile)
		for row in my_reader:
			row = [float(i) for i in row]
			Test_Y.append(row[0])
			Test_X.append(row[1:])



	return Train_X, Train_Y, Test_X, Test_Y


def uncertainty(Data, Labels):
	total_zero = 0.0;
	for i in range(len(Data)):
		if (Labels[i] == 0):
			total_zero += 1
	zero_portion = 0.0
	if (len(Data)):
		zero_portion = total_zero /  float(len(Data))
	
	one_portion = 1 - zero_portion
	u = 0
	if (zero_portion != 0):
		u = u -(zero_portion * math.log(zero_portion, 2))
	if (one_portion != 0):
		u = u - (one_portion * math.log(one_portion, 2))
	return u

def correct_num(Labels):
	zero = 0
	one = 0
	for l in Labels:
		if (l):
			one += 1
		else:
			zero += 1


	if (one >= zero):
		return one, 'one'
	else:
		return zero, 'zero'


def majority_label(Labels):
	zero = 0
	one = 0
	for l in Labels:
		if (l == 1):
			one += 1
		else:
			zero += 1


	if (one >= zero):
		return 1
	else:
		return 0



class Stump:
	def __init__(self):
		self.learned_info_gain = 0
		self.leaf_labels = [0, 0]
		self.decision_column = -1
	
	def learn_stump(self, Data, Labels, Column=None):
		total_uncertainty = uncertainty(Data, Labels)
	
		if (Column == None):
			gain = []
			learned_leaf_labels = []
			for Column in range(len(Data[0])):
				branch_zero = []
				branch_zero_labels = []
				
				branch_one = []
				branch_one_labels = []

				for i in range(len(Data)):
					if (Data[i][Column] == 0):
						branch_zero.append(Data[i])
						branch_zero_labels.append(Labels[i])
					else:
						branch_one.append(Data[i])
						branch_one_labels.append(Labels[i])

				zero_portion = float(len(branch_zero)) / float(len(Data))
				one_portion = 1 - zero_portion

				zero_u = uncertainty(branch_zero, branch_zero_labels)
				one_u = uncertainty(branch_one, branch_one_labels)
				
				info_gain = total_uncertainty - (zero_portion * zero_u) - (one_portion * one_u)

				learned_labels = [0, 0]
				learned_labels[0] = majority_label(branch_zero_labels)
				learned_labels[1] = majority_label(branch_one_labels)
				
				gain.append(info_gain)
				learned_leaf_labels.append(learned_labels)
			
			self.learned_info_gain = max(gain)
			self.decision_column = gain.index(self.learned_info_gain)
			
			self.leaf_labels[0] = learned_leaf_labels[self.decision_column][0]
			self.leaf_labels[1] = learned_leaf_labels[self.decision_column][1]
		else:
			branch_zero = []
			branch_zero_labels = []
			
			branch_one = []
			branch_one_labels = []

			for i in range(len(Data)):
				if (Data[i][Column] == 0):
					branch_zero.append(Data[i])
					branch_zero_labels.append(Labels[i])
				else:
					branch_one.append(Data[i])
					branch_one_labels.append(Labels[i])

			zero_portion = float(len(branch_zero)) / float(len(Data))
			one_portion = 1 - zero_portion

			zero_u = uncertainty(branch_zero, branch_zero_labels)
			one_u = uncertainty(branch_one, branch_one_labels)
			
			self.learned_info_gain = total_uncertainty - (zero_portion * zero_u) - (one_portion * one_u)

			self.leaf_labels[0] = majority_label(branch_zero_labels)
			self.leaf_labels[1] = majority_label(branch_one_labels)
			
			self.decision_column = Column
	
		
	def test_accuracy(self, Data, Labels):
		predicted_labels = []
		correct = 0.0
		for i in range(len(Labels)):
			if (Data[i][self.decision_column]):
				predicted_labels.append(self.leaf_labels[1])
				if (Labels[i] == self.leaf_labels[1]):
					correct += 1
			else:
				predicted_labels.append(self.leaf_labels[0])
				if (Labels[i] == self.leaf_labels[0]):
					correct += 1
		acc = correct / float(len(Labels))
		return acc, predicted_labels







#Added outputs zero_correct, label_zero, one_correct, label_one.  zero_correct, label_zero infers the feature is 0 and the label is the maximum correct targets in it.#
def info_gain_acc(Data, Labels, Column):
	total_uncertainty = uncertainty(Data, Labels)
	
	branch_zero = []
	branch_zero_labels = []
	
	branch_one = []
	branch_one_labels = []

	for i in range(len(Data)):
		if (Data[i][Column] == 0):
			branch_zero.append(Data[i])
			branch_zero_labels.append(Labels[i])
		else:
			branch_one.append(Data[i])
			branch_one_labels.append(Labels[i])

	zero_portion = float(len(branch_zero)) / float(len(Data))
	one_portion = 1 - zero_portion

	zero_u = uncertainty(branch_zero, branch_zero_labels)
	one_u = uncertainty(branch_one, branch_one_labels)

	zero_correct, label_zero = correct_num(branch_zero_labels)
	one_correct, label_one = correct_num(branch_one_labels)
	acc = float(zero_correct + one_correct) / float(len(Data))

	gain = total_uncertainty - (zero_portion * zero_u) - (one_portion * one_u)
	#print zero_correct, label_zero, one_correct, label_one
	
	return gain, acc, zero_correct, label_zero, one_correct, label_one


def Bootstrap_Sampling(Train_X, Train_Y, size):
	Boot_StrapsX = []
	Boot_StrapsY = []

	for z in range(0, len(Train_X)):
		Train_X[z].append(Train_Y[z])


	for x in range(0,size):
		SampleX = []
		SampleY = []
		for y in range(0, len(Train_X)):
			Current_Sample = cp.deepcopy(random.choice(Train_X))
			SampleY.append(Current_Sample[-1])
			Current_Sample.pop()
			SampleX.append(Current_Sample)
			
			

		Boot_StrapsX.append(SampleX)
		Boot_StrapsY.append(SampleY)
	return Boot_StrapsX, Boot_StrapsY

def Better_Bootstrap(Train_X, Train_Y, size):
	Boot_StrapsX = []
	Boot_StrapsY = []

	Full_Sample = cp.deepcopy(Train_X)
	for z in range(0, len(Train_X)):
		Full_Sample[z].append(Train_Y[z])


	for x in range(0,size):
		SampleX = []
		SampleY = []
		for y in range(0, len(Train_X)):
			Current_Sample = cp.deepcopy(random.choice(Full_Sample))
			SampleY.append(Current_Sample[-1])
			Current_Sample.pop()
			SampleX.append(Current_Sample)
			
			

		Boot_StrapsX.append(SampleX)
		Boot_StrapsY.append(SampleY)
	return Boot_StrapsX, Boot_StrapsY	
	
def Correct_Stump(Train_DataX, Train_DataY, Test_X, Test_Y):	
	Gains = []
	Target_Guess = []
	for c in range(len(Train_DataX[0])):

		info, train_acc, zero_correct, label_zero, one_correct, label_one = info_gain_acc(Train_DataX, Train_DataY, c)
		Gains.append(info)
	max_value = max(Gains)
	max_index = Gains.index(max_value)


	Gain, Accuracy, zero_correct, label_zero, one_correct, label_one = info_gain_acc(Test_X, Test_Y, max_index)
	
	for x in range(0, len(Test_Y)):
		if Test_X[x][max_index] == 1:
			if label_one == 'zero':
				Target_Guess.append(0)
			else:
				Target_Guess.append(1)
		else:
			if label_zero == 'one':
				Target_Guess.append(1)
			else:
				Target_Guess.append(0)
	
	return Target_Guess
	


def problem_1(Train_X, Train_Y, Test_X, Test_Y):

	d_stump = Stump()
	for c in range(len(Train_X[0])):
		d_stump.learn_stump(Train_X, Train_Y, c)
		info = d_stump.learned_info_gain
		acc = d_stump.test_accuracy(Test_X, Test_Y)[0]
		train_acc = d_stump.test_accuracy(Train_X, Train_Y)[0]
		print("%f \t%f" % (info, acc))
	
	my_stump.learn_stump(Train_X, Train_Y)
	info = my_stump.learned_info_gain
	acc = my_stump.test_accuracy(Test_X, Test_Y)[0]
	print("%f \t%f" % (info, acc))


def problem_2Bagging(Train_X, Train_Y, Test_X, Test_Y, size):	
	Boot_StrapsX, Boot_StrapsY = Bootstrap_Sampling(Train_X, Train_Y, size)	
	Bootstrapped_TargetListTEST = []
	Bootstrapped_TargetListTRAIN = []
	Predicted_TargetsTEST = []
	Predicted_TargetsTRAIN = []
	for x in range(0, size):

		Current_Targets = Correct_Stump(Boot_StrapsX[x], Boot_StrapsY[x], Test_X, Test_Y)
		Bootstrapped_TargetListTEST.append(Current_Targets)
	
	for y in range(0, size):

		Current_Targets = Correct_Stump(Boot_StrapsX[y], Boot_StrapsY[y], Train_X, Train_Y)
		Bootstrapped_TargetListTRAIN.append(Current_Targets)
	

	for z in range(0, len(Test_Y)):
		Testcount_1 = 0
		Testcount_0 = 0
		for s in range(0, size):
			if Bootstrapped_TargetListTEST[s][z] == 1:
				Testcount_1 += 1
			else:
				Testcount_0 += 1
				
		if Testcount_1 >= Testcount_0:
			Predicted_TargetsTEST.append(1)
		else:
			Predicted_TargetsTEST.append(0)
			

	
	for a in range(0, len(Train_Y)):
		Traincount_1 = 0
		Traincount_0 = 0
		for b in range(0, size):
			if Bootstrapped_TargetListTRAIN[b][a] == 1:
				Traincount_1 += 1
			else:
				Traincount_0 += 1
				
		if Traincount_1 >= Traincount_0:
			Predicted_TargetsTRAIN.append(1)
		else:
			Predicted_TargetsTRAIN.append(0)
			
	count_Test = 0
	for e in range(0, len(Test_Y)):
		if Test_Y[e] == Predicted_TargetsTEST[e]:
			count_Test += 1
	print count_Test, len(Test_Y)
		
	count_Train = 0
	for r in range(0, len(Train_Y)):
		if Train_Y[r] == Predicted_TargetsTRAIN[r]:
			count_Train += 1
	print count_Train, len(Train_Y)		

def voted_predict(predictions, size):
	threshold = size / 2.0
	voted_guess = []
	
	for sample in range(len(predictions[0])):
		guess = 0.0
		for pred in range(size):
			guess += predictions[pred][sample]
		
		if (guess > threshold):
			voted_guess.append(1)
		else:
			voted_guess.append(0)
	
	return voted_guess

def calc_acc(pred, label):
	correct = 0.0
	for i in range(len(pred)):
		if (pred[i] == label[i]):
			correct += 1
	
	return correct / float(len(pred))
	
def problem_2(Train_X, Train_Y, Test_X, Test_Y):
	test_accuracies = []
	train_accuracies = []
	bag_sizes = [5, 10, 15, 20, 25, 30]
	for trial_run in range(10):
		Bootstrap_X, Bootstrap_Y = Better_Bootstrap(Train_X, Train_Y, 30)	
		stumps = []
		for i in range(30):
			s = Stump()
			s.learn_stump(Bootstrap_X[i], Bootstrap_Y[i])
			stumps.append(s)
			print(s.test_accuracy(Test_X, Test_Y)[0])

		all_train_predictions = []
		all_test_predictions = []
	
		for i in range(30):
			all_train_predictions.append(stumps[i].test_accuracy(Train_X, Train_Y)[1])
			all_test_predictions.append(stumps[i].test_accuracy(Test_X, Test_Y)[1])
	
		bag_sizes = [5, 10, 15, 20, 25, 30]
		train_bag_predictions = []
		test_bag_predictions = []
		for b in bag_sizes:
			train_bag_predictions.append(voted_predict(all_train_predictions, b))
			test_bag_predictions.append(voted_predict(all_test_predictions, b))
	
		train_bag_accuracy = []
		test_bag_accuracy = []
		for i in range(len(bag_sizes)):
			train_bag_accuracy.append(calc_acc(train_bag_predictions[i], Train_Y))
			test_bag_accuracy.append(calc_acc(test_bag_predictions[i], Test_Y))
	
		for i in range(len(bag_sizes)):
			print ("%f\t%f" % (train_bag_accuracy[i], test_bag_accuracy[i]))
		
		test_accuracies.append(test_bag_accuracy)
		train_accuracies.append(train_bag_accuracy)

	average_train_accuracy = []
	average_test_accuracy = []

	for bag_num in range(len(test_accuracies[0])):
		train_avg = 0.0
		test_avg = 0.0		
		for i in range(10):
			train_avg += train_accuracies[i][bag_num]
			test_avg += test_accuracies[i][bag_num]

		average_train_accuracy.append(train_avg / 10.0)
		average_test_accuracy.append(test_avg / 10.0)

	for t in average_test_accuracy:
		print(t)

	for t in average_train_accuracy:
		print(t)
	
	plt.plot(bag_sizes, average_train_accuracy, label='Train Accuracy')
	plt.plot(bag_sizes, average_test_accuracy, label='Test Accuracy')
	plt.legend(loc='best')
	plt.xlabel("Number of Decision Stumps")
	plt.ylabel("Accuracy")
	plt.title("Accuracy of Bagged Decision Stumps")
	
	plt.savefig("problem_2_plot.png")
	plt.show()
	


		
	
def stump_accuracy(Data, Labels, Column):
	pass
	
	
	
	

def main():
	size = 5
	Train_X, Train_Y, Test_X, Test_Y = read_csv_data("SPECT-train.csv", "SPECT-test.csv")
	
	print(len(Train_X))
	
	print(len(Train_Y))
	print(len(Test_X))
	print(len(Test_Y))

	#problem_1(Train_X, Train_Y, Test_X, Test_Y)
	#problem_2Bagging(Train_X, Train_Y, Test_X, Test_Y, size)

	problem_2(Train_X, Train_Y, Test_X, Test_Y)
if __name__ == "__main__":
	main()
