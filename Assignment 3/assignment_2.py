from __future__ import division
import csv
import math
import random
import copy as cp


















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

	for c in range(len(Train_X[0])):
		info, train_acc, zero_correct, label_zero, one_correct, label_one = info_gain_acc(Train_X, Train_Y, c)
		who_cares, test_acc,  zero_correct, label_zero, one_correct, label_one = info_gain_acc(Test_X, Test_Y, c)
		print("%f \t%f" % (who_cares, test_acc))


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
	
def stump_accuracy(Data, Labels, Column):
	pass
	
	
	
	

def main():
	size = 5
	Train_X, Train_Y, Test_X, Test_Y = read_csv_data("SPECT-train.csv", "SPECT-test.csv")
	
	print(len(Train_X))
	
	print(len(Train_Y))
	print(len(Test_X))
	print(len(Test_Y))

	problem_1(Train_X, Train_Y, Test_X, Test_Y)
	problem_2Bagging(Train_X, Train_Y, Test_X, Test_Y, size)

if __name__ == "__main__":
	main()
