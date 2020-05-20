# Framework
import torch
import math
import sys

torch.set_grad_enabled(False)

# Basic class to build linear layer

class Linear:

	def __init__(self,dim_input,dim_output,init_type):
		
		self.dim_output = dim_output

		# Initialization of weights and biases of the layer
		if(init_type == 'He'):
			C = math.sqrt(2/dim_input)
		else:
			C = math.sqrt(1/dim_input)


		self.weights = torch.empty(dim_output, dim_input).normal_()*C
		self.biases = torch.empty(dim_output,1).normal_()*C
		
	def update_weights(self,learning_rate):

		if(learning_rate < 0):
			print("Learning rate need to be superior to 0 !")
			sys.exit(1)

		self.weights = self.weights - learning_rate*self.grad_weights
		self.biases = self.biases - learning_rate*self.grad_bias


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
		  return torch.mm(torch.t(gradient),self.weights)

	def param(self):
		return  self.weights, self.biases, self.local_grad

# Activation functions classes 

class Relu:

	def forward(self, x):
		self.activated = torch.clamp(x, min=0.0)
		return self.activated
		

	def  backward(self , gradient):
		
		return 1.0*gradient if self.activated > 0 else 0.0

	def  param(self):
		return  []

class Tanh:

	def f(self,x):

		return torch.tanh(x)

	def forward(self, x):
		self.activated = self.f(x)
		return self.activated

	def  backward(self , gradient):

		return gradient*(1-self.f(self.activated)**2)

	def  param(self):
		return  []

class Sigmoid:

	def f(self, x):
		return 1/(1 + torch.exp(-x))

	def forward(self, x):
		self.activated = self.f(x)
		return self.activated

	def backward(self, gradient):
		return gradient*torch.mm(self.activated, (1 - self.activated)) 

	def param(self):
		return []

# Main class call to build the network

class Sequential:

	def __init__(self,input_size,output_size,hidden_sizes,list_activ_function,loss_function):

		# Verification of the inputs
		if(hidden_sizes.shape[0] != len(list_activ_function)):
			print("Error: We need ONE activation function for each hidden layer output!")
			sys.exit(1)

		for i in range(len(list_activ_function)):
			if(list_activ_function[i] != 'Relu' and list_activ_function[i] != 'Tanh' and list_activ_function[i] != 'Sigmoid'):
				print('Error: Activation function allow are Tanh, Relu and Sigmoid !')
				sys.exit(1)

		if(loss_function != 'MSE'):
			print('Error: MSE is the only loss function allowed !')
			sys.exit(1)

		# Assignation of the inputs
		self.net_input_size = input_size
		self.net_output_size = output_size
		self.dim_hidden = hidden_sizes
		self.list_activ_string = list_activ_function # Easier to check string than object for comparison

		self.activ_functions = []
		for act in list_activ_function:
			if(act == 'Relu'):
				self.activ_functions.append(Relu())
			elif(act == 'Tanh'):
				self.activ_functions.append(Tanh())

		if(loss_function == 'MSE'):
			self.loss = LossMSE()

		# Print the network structure
		print("\nInput: size:  ",self.net_input_size)
		for i in range(self.dim_hidden.shape[0]):
			print("Hidden layer: ","size: ",self.dim_hidden[i].item(),",Activation function: ",self.list_activ_string[i],'-')
		print("Ouput: size:  ",self.net_output_size,',Loss criterion:',loss_function,'\n')

		self.build_network()



	def __call__(self,input,labels):
		print('Start forward pass\n' )
		self.forward(input,labels)

	def build_network(self):
		self.network = []

		init_type = self.choose_init(0)

		self.network.append(Linear(self.net_input_size,self.dim_hidden[0],init_type))
		self.network.append(self.activ_functions[0])

		for layer in range(self.dim_hidden.shape[0]-2):
			init_type = self.choose_init(layer)
			self.network.append(Linear(self.dim_hidden[layer],self.dim_hidden[layer+1],init_type))
			self.network.append(self.activ_functions[layer+1])

		init_type = self.choose_init(-1)
		self.network.append(Linear(self.dim_hidden[-1],self.net_output_size,init_type))
		self.network.append(self.activ_functions[-1])

		#print(self.network)

	def choose_init(self,layer):

		if(self.list_activ_string[layer] == 'Relu'):
			init_type = 'He'
		else:
			init_type = 'Xavier'
		
		return init_type

	def forward(self, x, y):

		if(x.shape[0] != self.net_input_size):
			print('Error: Not right input size for this network')

		for net in self.network:
			x = net.forward(x)

		self.loss.computeMSE(y,x)
		print('loss = ',self.loss)

		return self.loss
	
	def  backward(self):
		z = self.loss.backward()
		for net in reversed(self.network):
			z = net.backward(z)

	def  param(self):
		return []

# Loss Functions

class LossMSE:
	
	def computeMSE(self, y, y_pred):
		# last step of the forward pass 
		MSE = (y_pred-y).pow(2).sum() # this is not the MEAN square error (but just the square error) 
		# actually it's correct because we compute it for one sample
		self.y = y
		self.y_pred = y_pred
		return MSE

	def backward(self,):
		num_samples = self.y.shape[0]
		return num_samples*2*(self.y_pred-self.y)


class LossCrossEntropy:

	def forward(self, *input):
		raise  NotImplementedError
	def  backward(self , *gradwrtoutput):
		raise  NotImplementedError
	def  param(self):
		return  []

