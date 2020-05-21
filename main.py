import torch
import sys
import math
import framework

def main():
	project_2()

def project_2():
	torch.set_grad_enabled(False)

	MINI_BATCH = 1
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
	nb_test_errors = compute_nb_errors(net,test_set,test_labels,MINI_BATCH)

	print('Test error for our network : {:0.2f}%% {:d}/{:d}'.format((100 * nb_test_errors) / test_set.size(1),
                                                   nb_test_errors, test_set.size(1)))


	return []

def train_model(network, train_input, train_target, mini_batch_size, nb_epoch):

	for e in range(nb_epoch):
		sum_loss = 0
		for b in range(0, train_input.size(1), mini_batch_size):
			output = network.forward(train_input[:,b:b+mini_batch_size])		# Compute forward pass and loss
			loss = network.loss_criterion(output,train_target[:,b:b+mini_batch_size])
			network.backward() # Compute backward pass and update the weights and biases
			print("Loss = ",loss)
			print("sum_loss = ", sum_loss)
			sum_loss = sum_loss + loss

		print(e,sum_loss)

def compute_nb_errors(network, input, target, mini_batch_size):
	nb_errors = 0
	for b in range(0,input.size(0), mini_batch_size):
		output = network.forward(input[:,b:b+mini_batch_size])
		print('output =', output)
		print('target =', target)
		
		for k in range(mini_batch_size):
			if ( not torch.all(torch.eq(target[0,b+k],output[:,k]))):
				nb_errors = nb_errors + 1
	return nb_errors





if __name__ == "__main__":
    main()
