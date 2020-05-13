import torch
import sys
import math
import framework

def main():
	
	project_1()

	project_2()


def project_1():
	return []

def project_2():

	# Data set creation

	N = 1000 # Number of points

	training_set = torch.rand(2,N)
	test_set = torch.rand(2,N)

	train_labels = torch.empty(2,N)
	test_labels = torch.empty(2,N)

	for i in range(N):
		if math.sqrt( (training_set[0,i] - 0.5)**2 + (training_set[1,i] - 0.5)**2 ) < 1/math.sqrt(2*math.pi):
			train_labels[0,i] = 1
			train_labels[1,i] = 0
		else: 
			train_labels[0,i] = 0
			train_labels[1,i] = 1

		if math.sqrt( (test_set[0,i] - 0.5)**2 + (test_set[1,i] - 0.5)**2 ) < 1/math.sqrt(2*math.pi):
			test_labels[0,i] = 1
			test_labels[1,i] = 0
		else: 
			test_labels[0,i] = 0
			test_labels[1,i] = 1


	# Test
	hidden = torch.tensor([25,25,25])
	net = framework.Sequential(2,2,hidden,['Relu','Relu','Tanh'],'MSE')

	net(training_set,train_labels)

	# Training


	return []

def train_model(network, train_input, train_target, mini_batch_size, nb_epoch):

	for e in range(nb_epoch):
		sum_loss = 0
		for b in range(0, train_input.size(0), mini_batch_size):
			loss = network.forward(train_input,train_target)		# Compute forward pass and loss
			network.backward()										# Compute backward pass and update the weight and biases
			sum_loss = sum_loss + loss.item()
		print(e,sum_loss)



if __name__ == "__main__":
    main()
