# Framework
import torch
import math
import sys

torch.set_grad_enabled(False)

# Basic class to build a linear layer

class Linear:

	def __init__(self,dim_input,dim_output,init_type,learning_rate):
		
		
		self.dim_output = dim_output

		# Initialization of the weights and biases of the layer
		if(init_type == 'He'):
			C = math.sqrt(2/dim_input.item())
		else:
			C = math.sqrt(1/dim_input.item())
		
		self.weights = torch.empty(dim_output, dim_input).normal_(mean=0, std=1)*C
		self.biases = torch.empty(dim_output,1).zero_()

		self.grad_weights = torch.empty(dim_output,dim_input).zero_()
		self.grad_bias = torch.empty(dim_output,1).zero_()

		self.learning_rate = learning_rate

	def update_weights(self):

		self.weights = self.weights - self.learning_rate*self.grad_weights
		self.biases = self.biases - self.learning_rate*self.grad_bias


	def forward(self,input):
		# Forward Pass. Computes the output of the linear layer and the local gradient.
		self.input = input
		z = torch.mm(self.weights,input) + self.biases
		return z

	def backward(self , gradient):
		
		# Gradient with respect to weights
		self.grad_weights = torch.mm(gradient,torch.t(self.input))
		# Gradient with respect to bias
		self.grad_bias = torch.sum(gradient)
		# Global gradient, to be propagated backwards
		self.grad_global = torch.mm(torch.t(self.weights),gradient)
		self.update_weights()
		return self.grad_global


	def param(self):
		return  self.weights, self.grad_weights, self.biases, self.grad_bias

# Activation functions classes 

class Relu:

	def forward(self, x):
		self.input = x
		return torch.clamp(x, min=0.0)
		

	def  backward(self , gradient):
		self.input[self.input >= 0] = 1
		self.input[self.input < 0] = 0
		return self.input * gradient

	def  param(self):
		return  []

class Tanh:

	def f(self,x):

		return torch.tanh(x)

	def forward(self, x):
		self.activated = self.f(x)
		return self.activated

	def  backward(self , gradient):

		return gradient*(1-self.activated**2)

	def  param(self):
		return  []

class Sigmoid:

	def f(self, x):
		return 1/(1 + torch.exp(-x))

	def forward(self, x):
		self.activated = self.f(x)
		return self.activated

	def backward(self, gradient):

		return gradient*(self.activated*(1 - self.activated)) 

	def param(self):
		return []

# Main class called to build the network

class Sequential:

	def __init__(self,input_size,output_size,hidden_sizes,list_activ_function,loss_function,learning_rate):

		# Verification of the inputs
		if(hidden_sizes.shape[0] != len(list_activ_function)-1):
			print("Error: We need ONE activation function for each hidden layer output!")
			sys.exit(1)

		for i in range(len(list_activ_function)):
			if(list_activ_function[i] != 'Relu' and list_activ_function[i] != 'Tanh' and list_activ_function[i] != 'Sigmoid'):
				print('Error: Activation function allow are Tanh, Relu and Sigmoid !')
				sys.exit(1)

		if(loss_function != 'MSE'):
			print('Error: MSE is the only loss function allowed !')
			sys.exit(1)

		if(learning_rate < 0):
			print("Learning rate need to be superior to 0 !")
			sys.exit(1)

		# Assignation of the inputs
		self.net_input_size = torch.tensor([input_size])
		self.net_output_size = output_size
		self.dim_hidden = hidden_sizes
		self.list_activ_string = list_activ_function # Easier to check string than object for comparison

		self.activ_functions = []
		for act in list_activ_function:
			if(act == 'Relu'):
				self.activ_functions.append(Relu())
			elif(act == 'Tanh'):
				self.activ_functions.append(Tanh())
			elif(act == 'Sigmoid'):
				self.activ_functions.append(Sigmoid())

		if(loss_function == 'MSE'):
			self.loss = LossMSE()

		self.learning_rate = learning_rate

		self.build_network()


	def build_network(self):
		self.network = []

		init_type = self.choose_init(0)

		self.network.append(Linear(self.net_input_size,self.dim_hidden[0],init_type,self.learning_rate))
		self.network.append(self.activ_functions[0])

		for layer in range(self.dim_hidden.shape[0]-1):
			init_type = self.choose_init(layer)
			self.network.append(Linear(self.dim_hidden[layer],self.dim_hidden[layer+1],init_type,self.learning_rate))
			self.network.append(self.activ_functions[layer+1])

		init_type = self.choose_init(-1)
		self.network.append(Linear(self.dim_hidden[-1],self.net_output_size,init_type,self.learning_rate))
		self.network.append(self.activ_functions[-1])
		
		#print(self.network)

	def choose_init(self,layer):

		if(self.list_activ_string[layer] == 'Relu'):
			init_type = 'He'
		else:
			init_type = 'Xavier'
		
		return init_type

	def forward(self, x):

		if(x.shape[0] != self.net_input_size):
			print('Error: Not right input size for this network')

		for net in self.network:
			x = net.forward(x)
	
		return x

	def loss_criterion(self, x, y):
		self.loss.computeMSE(y,x)
		return self.loss.param()
	
	def  backward(self):
		
		z = self.loss.backward()

		for net in reversed(self.network):
			z = net.backward(z)

	def  param(self):
		return self.network

# Loss Function

class LossMSE:
	
	def computeMSE(self, y, y_pred):
		# last step of the forward pass 
		self.number_elem = y.shape[0]*y.shape[1]
		self.mse = (1/self.number_elem)*torch.pow(y_pred-y,2).sum() 
		self.y = y
		self.y_pred = y_pred
		return self.mse

	def backward(self):

		return (1/self.number_elem)*2*(self.y_pred-self.y)

	def param(self):
		return self.mse


