# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:19:25 2018

@author: Skeletor
"""

#implementing a basic neural network,feel free to use this as a example.
#pls no nitpick,I put this together after reading the pytorch docs for a bit.
#If you're in visualstudio code and pycharm and are getting uncallable errors,there's a link explaining why
#https://discuss.pytorch.org/t/v0-4-pycharm-unknown-inspections-unexpected-argument-not-callable-etc/17810

import torch              #here's a obvious import of stuff that's needed
import torch.nn as nn

"I'm using the neural network to predict grades in relation to ammount of hours studied,and ammount of time spent sleeping, sth is shorthand for study hours,and hours sleeping is going to be slp"
"So it's formatted [sth,slp] in tensor A,which is where the hours studied,and hours sleeping will be."                                   

A = torch.tensor (([3,8], [1,4], [2,6], [6,12]), dtype=torch.float)  #torch.float is a 32x floating point,and tensor is creating a multi-dimensional matrix with elements of a single data type,as opposed to torch.storage,which is 1d.
B = torch.tensor (([80], [50], [78], [98]), dtype=torch.float) #this is a 4x1 tensor,for obvious reasons.  Look above for the definition of tensor
predicta = torch.tensor (([3,6]), dtype=torch.float)#this is a prediction input for a grade using the sth and slp parameters that the network learned. It's not accurate

#Now,we're going to scale the sample data a bit

A_max,_= torch.max(A,0)     #max finds the max value of a tensor
predicta_max,_= torch.max(predicta,0)

A = torch.div(A,A_max)  #div divides 2 tensors
predicta = torch.div(predicta,predicta_max)
B = B/100 

#Now that the data has been processed, here's where the fun begins.

class neural_network(nn.Module):
    def __init__(self,):
        super(neural_network,self).__init__()
        #params and network structure
        self.inputSize = 2  
        self.outputSize = 1
        self.hiddenSize = 3              
        #weights.  Both weight matrixes are initalized with pesudo-random values,using torch.radn The weight,is in essence,a denotation of how much a feauture matters in a neural model,or mathematically,a linear regression of all outputs in the previous layer
        self.weight1 = torch.randn(self.inputSize, self.hiddenSize)  #this stores params from input to hidden
        self.weight2 = torch.randn(self.hiddenSize, self.outputSize) #this stores from hidden to output
        #No bias is inclued in this model as yet,just keeping it simple.
        
    def sigmoid(self, s): # Sigmoid stands for a signmoid neuron,which introduces non-linearity into this model,and is a activation function for the artificial neurons
        return 1 / (1 + torch.exp(-s)) #This differs from a basic perceptron, (perceptron example coming soon) because instead of a discrete 1 or 0 value,that the perceptron gives,a sigmoid function/neuron gives a more smooth range of values
    
    def sigmoidPrime(self, s):# derivative of sigmoid
        return s * (1 - s)
        
    def forward(self, A): #this foward pass is where the magic happens,the data is fed into the computation graph
        self.z = torch.matmul(A, self.weight1) #matmul is the matrix product of 2 tensors. Behavior depends on dimensionality
        self.z2 = self.sigmoid(self.z)#activation function. 
        self.z3 = torch.matmul(self.z2, self.weight2)
        r = self.sigmoid(self.z3) #final activation function
        return r
    
    
    def backwards(self, A,B,r): #since the foward pass is where the data is fed,the backwards pass counts the changes in weights,using the gradient descend algorithm. This is (defacto) learning           
        self.r_error = B-r #error in output
        self.r_delta = self.r_error * self.sigmoidPrime(r)
        self.z2_error = torch.matmul(self.r_delta,torch.t(self.weight2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.weight1 += torch.matmul(torch.t(A), self.z2_delta)
        self.weight2 += torch.matmul(torch.t(self.z2), self.r_delta)  
        
        #this block of code is matrix multiplactions,transposition operations,and gradient descent.
        
        #all that's left to do now is training of this basic network.
    def train(self, A,B): #fowards and backwards pass for training
        r = self.forward(A)
        self.backwards(A, B, r)
        
        
    def saveweights(self, model):
        torch.save(model, "basic")
        #model can be reloaded with torch.load("simple")
        
        
    def predict(self):
        print ("Predicted data based off trained weights: ")
        print ("Input (scaled): \n" + str(predicta))
        print ("Output: \n" + str(self.forward(predicta))) 
    #this is pretty self-explainitory
    
  
basic = neural_network() #instance created of the network
for i in range(1000):
    print ("#" + str(i) + " Loss: " + str(torch.mean((B - basic(A))**2).detach().item()))  #Mean sum,squared loss.  torch.mean is self explainitory,but torch.detach creates a tensor that shares storage with a tensor that doesn't require grad
    basic.train(A, B)
basic.saveweights(basic)
basic.predict()
#observe loss steadily decrease as the network learns

