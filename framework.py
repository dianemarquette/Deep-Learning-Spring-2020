# Framework
import torch
import math
import sys

torch.set_grad_enabled(False)

class Linear(object):

	def __init__(self,input,dim_output):
		
		self.dim_output = dim_output

		# Initialization of weights and biases of the layer
		self.weights = torch.empty(dim_output, input.size()).normal_()
		self.biases = torch.empty(dim_output,1).normal_()
		return []

	def forward(self,input):
		# Forward Pass. Computes the output of the linear layer and the local gradient.
		z = torch.mm(self.weights,self.input) + self.biases
		self.local_grad = self.weights
		return z

	def  backward(self , *gradwrtoutput):
		raise  NotImplementedError

	def  param(self):
		return  self.weights, self.biases, self.local_grad

class Relu(object):

	def forward(self, x):
		return torch.clamp(x, min=0.0)
		

	def  backward(self , z):
		
		return (z > 0) * 1.0

	def  param(self):
		return  []

class Tanh(object):

	def f(x):
		return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

	def forward(self, x):
		x = input
		return f(x)

	def  backward(self , z):
		 
		return (1-f(z)**2)

	def  param(self):
		return  []

class Sequential(object):

	def __init__(self,input_size,output_size,hidden_sizes,list_activ_function,loss_function):

		# Verification of the inputs
		if(hidden_sizes.shape[0] != len(list_activ_function)):
			print("Error: We need ONE activation function for each hidden layer output!")
			sys.exit(1)

		for i in range(len(list_activ_function)):
			if(list_activ_function[i] != 'Relu' and list_activ_function[i] != 'Tanh'):
				print('Error: Activation function allow are Tanh and Relu !')
				sys.exit(1)

		if(loss_function != 'MSE'):
			print('Error: Loss function allow are MSE !')

		# Assignation of the inputs
		self.net_input_size = input_size
		self.net_output_size = output_size
		self.dim_hidden = hidden_sizes

		self.activ_functions = []
		for act in list_activ_function:
			if(act == 'Relu'):
				self.activ_functions.append(Relu())
			elif(act == 'Tanh'):
				self.activ_functions.append(Tanh())

		if(loss_function == 'MSE'):
			self.loss = LossMSE()

		# Print the network structure
		print("\nInput: ",self.net_input_size)
		for i in range(self.dim_hidden.shape[0]):
			print("Hidden layer: ",i," ,size: ",self.dim_hidden[i].item(),",Activation function: ",self.activ_functions[i],'-')
		print("Ouput: ",self.net_output_size,'\n')

		self.init_weight_bias()

	def __call__(self,input):
		print('Start forward pass\n' )
		#self.forward()


	def forward(self, x):
		raise  NotImplementedError
	def  backward(self , z):
		raise  NotImplementedError
	def  param(self):
		print(self.all_weights)


class LossMSE(object):

	class LossMSE(object):

	def computeMSE(self, y, y_pred):
		MSE = (y_pred-y).pow(2).sum()


class LossCrossEntropy():

	def forward(self, *input):
		raise  NotImplementedError
	def  backward(self , *gradwrtoutput):
		raise  NotImplementedError
	def  param(self):
		return  []

