import torch
import sys
import math
import framework


def main():
	project_2()

def project_2():
	torch.set_grad_enabled(False)

	MINI_BATCH = 1
	EPOCH = 25

	eta = 1e-2   # Learning rate

	# Data set generation

	N = 1000 # Number of points

	train_input = torch.rand(N,2)
	test_input = torch.rand(N,2)

	train_labels = torch.empty(N,2)
	test_labels = torch.empty(N,2)

	for i in range(N):
		if math.sqrt( (train_input[i,0] - 0.5)**2 + (train_input[i,1] - 0.5)**2 ) < 1/math.sqrt(2*math.pi):
			train_labels[i,0] = 1
			train_labels[i,1] = 0
		else: 
			train_labels[i,0] = 0
			train_labels[i,1] = 1

		if math.sqrt( (test_input[i,0] - 0.5)**2 + (test_input[i,1] - 0.5)**2 ) < 1/math.sqrt(2*math.pi):
			test_labels[i,0] = 1
			test_labels[i,1] = 0
		else: 
			test_labels[i,0] = 0
			test_labels[i,1] = 1

	# Normalization
	mean, std = train_input.mean(), train_input.std()
	train_input.sub_(mean).div_(std)
	test_input.sub_(mean).div_(std)

	hidden = torch.tensor([25,25,25])
	net = framework.Sequential(2,2,hidden,['Relu','Relu','Relu','Tanh'],'MSE',eta)

	# Training
	train_model(net, train_input, train_labels, MINI_BATCH,EPOCH)
	# Testing
	print("Training error :",compute_nb_errors(net, train_input, train_labels,MINI_BATCH) / train_input.size(0)*100,'%')
	print('Testing error :',compute_nb_errors(net, test_input, test_labels,MINI_BATCH) / test_input.size(0)*100,'%')

	return []

def train_model(network, train_input, train_target, mini_batch_size, nb_epoch):

	for e in range(nb_epoch):
		sum_loss = 0
		for b in range(0, train_input.size(0), mini_batch_size):
			output = network.forward(train_input[b,:].view(2,1))		# Compute forward pass and loss
			loss = network.loss_criterion(output,train_target[b,:].view(2,1))
			network.backward() # Compute backward pass and update the weights and biases

def compute_nb_errors(network, input, target, mini_batch_size):
	nb_errors = 0
	for b in range(0,input.size(0), mini_batch_size):
		output = network.forward(input[b,:].view(2,1))

		if ( target[b,:].max(0)[1] != output.max(0)[1]):
			nb_errors = nb_errors + 1
	return nb_errors

if __name__ == "__main__":
    main()
