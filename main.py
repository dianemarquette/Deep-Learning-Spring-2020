import torch
from torch import nn
import sys
import math
import framework

def main():
	
	project_1()

	project_2()


def project_1():
	return []

def project_2():

	N = 1000 # Number of points

	training_set = torch.rand(2,N)
	test_set = torch.rand(2,N)

	train_labels = torch.empty(N)
	test_labels = torch.empty(N)

	for i in range(N):
		if math.sqrt( (training_set[0,i] - 0.5)**2 + (training_set[1,i] - 0.5)**2 ) < 1/math.sqrt(2*math.pi):
			train_labels[i] = 1
		else: 
			train_labels[i] = 0

		if math.sqrt( (test_set[0,i] - 0.5)**2 + (test_set[1,i] - 0.5)**2 ) < 1/math.sqrt(2*math.pi):
			test_labels[i] = 1
		else: 
			test_labels[i] = 0


	# Test
	hidden = torch.tensor([25,25,25])
	net = framework.Sequential(2,2,hidden,['Relu','Relu','Tanh'],'MSE')

	return []


if __name__ == "__main__":
    main()