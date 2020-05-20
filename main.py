import torch
import sys
import math
import framework

def main():
	project_2()

def project_2():
	torch.set_grad_enabled(False)

	MINI_BATCH = 100
	EPOCK = 1

	eta = 1e-1  # Learning rate

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
	net = framework.Sequential(2,2,hidden,['Relu','Relu','Tanh'],'MSE',eta)

	net(training_set,train_labels)

	# Training
	train_model(net, training_set, train_labels, MINI_BATCH,EPOCK)
	# Testing
	nb_test_errors = compute_nb_errors(net,training_set,train_labels,MINI_BATCH)

	print('Test error for our network : {:0.2f}%% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                   nb_test_errors, test_input.size(0)))


	return []

def train_model(network, train_input, train_target, mini_batch_size, nb_epoch):

	for e in range(nb_epoch):
		sum_loss = 0
		for b in range(0, train_input.size(0), mini_batch_size):
			loss = network.forward(train_input,train_target)		# Compute forward pass and loss
			network.backward()										# Compute backward pass and update the weight and biases
			sum_loss = sum_loss + loss.item()
		print(e,sum_loss)

def compute_nb_errors(network, input, target, mini_batch_size):

	for b in range(0,input.size(0), mini_batch_size):
		output = network(input.narrow(0,b,mini_batch_size))
		_, predicted_classes = ouput.max(1)

		for k in range(mini_batch_size):
			if target[b+k] != predicted_classes[k]:
				nb_errors = nb_errors + 1
	return nb_errors





if __name__ == "__main__":
    main()
