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

	eta = 1e-1  / 1000  # Learning rate

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



	
	hidden = torch.tensor([25,25,25])
	net = framework.Sequential(2,2,hidden,['Relu','Relu'],'MSE',eta)
	# Training
	train_model(net, training_set, train_labels, MINI_BATCH,EPOCK)
	# Testing
	print('train_error {:.02f}% test_error {:.02f}%'.format(
        compute_nb_errors(net, training_set, train_labels,MINI_BATCH) / training_set.size(1) * 100,
        compute_nb_errors(net, test_set, test_labels,MINI_BATCH) / test_set.size(1) * 100
        )
        )


	return []

def train_model(network, train_input, train_target, mini_batch_size, nb_epoch):

	for e in range(nb_epoch):
		sum_loss = 0
		for b in range(0, train_input.size(1), mini_batch_size):
			output = network.forward(train_input[:,b].view(2,1))		# Compute forward pass and loss
			loss = network.loss_criterion(output,train_target[:,b].view(2,1))
			network.backward() # Compute backward pass and update the weights and biases

			sum_loss = sum_loss + loss

		print(e,sum_loss)

def compute_nb_errors(network, input, target, mini_batch_size):
	nb_errors = 0
	for b in range(0,input.size(1), mini_batch_size):
		output = network.forward(input[:,b].view(2,1))
		#print('output =',output, output.max(0))
		#print('label =',target[:,b],target[:,b].max(0))

		if ( target[0,b].max(0)[1] != output.max(0)[1]):
			nb_errors = nb_errors + 1
	return nb_errors





if __name__ == "__main__":
    main()
