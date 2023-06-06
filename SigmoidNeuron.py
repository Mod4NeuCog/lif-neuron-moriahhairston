#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:29:28 2023

Signmoid Function

@author: moriahlashaunhairston
"""


import numpy as np
import math 



#class SigmoidNeuron:
 #   def __init__(self, weights):#__init__ = what is this class going to take (weights will be the inputs given)
  #      self.weights = weights
   #     
    #def activity_level(self):
     #   return 1/(1+math.exp(-sum(self.weights))) #the reason we put -sum and not -x because it is a vector and we need the sum of the weights instead of the sum of one weight
        


#weights = [3,7, 8]
#neuron1 = SigmoidNeuron(weights)
#output = neuron1.activity_level() #this is calling the function.. the thing that is actually doing the work 

#print(output)

# print(SigmoidNeuron([2,3,4]).activity_level()) #could write it this way but it is easier to see errors the other way and it keeps the code clean.


#class Neuron:
#    def __init__(self,input):
#        self.input = input #initating and will change later
        
#    def totalweight(self, weights):
#        for i in range(len(weights)):
#            total_sum = weights[i] * self.input[i]
    
#    def sigmoid(x): ## how do i use this? 
#        1/(1+math.exp(-x))



#Define the inputs and weights
#inputs = [0.3, 0.1, 0.2]
#weights = [0, 0 , 0]

#class Layer: 
#    def __init__(self):




class GeneralNeuron:
    def __init__(self, inputs, threshold): #what it is
        self.inputs = inputs
        self.threshold = threshold
    
    def sumweights(self):
        a = sum(self.inputs) #what it does #because we are looking for the sum, it needs to be an object [] because we "can't" sum one number
        
        if a >= self.threshold:# always put self because it will change according to the inputs given
            print("I'm on fire Bitch!")
        else:
            print("light this Bitch up!")
            

#Step 1: Create the neuron used to access the function which is in the class 
#Neuron2 = GeneralNeuron([15, 40, 10], 70)
#Neuron3 = GeneralNeuron([82, 90], 76)
#Neuron4 = GeneralNeuron([99, 69], 10000)


#Step 2: access function (sumweights) to see if the bitch is on fire
#Neuron2.sumweights()
#Neuron3.sumweights()
#Neuron4.sumweights()



class SigmoidNeuron:
    def __init__(self, weights):#__init__ = what is this class going to take (weights will be the inputs given)
        self.weights = weights
        
    def activity_level(self):
        if self.weights == False: #we put this to false because we dont want it to calcule the first layer because it was already given in the input. 
            return 0
        else:
            return 1/(1+math.exp(-sum(self.weights))) #the reason we put -sum and not -x because it is a vector and we need the sum of the weights instead of the sum of one weight
        

# create an Integrate and Fire Neuron class  that is similar to theSigmoid neuron. IF also takes weights but also has a threshold and for the activity level, it calculates the sum of these weights and see if they pass the threshold. 
class IntergrateFireNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights 
        self.threshold = threshold
        
    def activity_level(self):
        if self.weights == False:
            return 0
        else:
            if sum(self.weights) >= self.threshold:
                return 1
            
            else:
                return 0 
        
## Add IF into the layer 

class Layer:
    def __init__(self, number_of_neurons, weights, neuron_threshold):
        self. number_of_neurons = number_of_neurons
        self.neurons = []
        self.weights = weights
        self.layer_activation = []
        self.neuron_threshold = neuron_threshold
      
    def create_neurons(self):
        
        #weights = [[4,6],[5,8,1]]#weights are static like this, meaning they can't change
        for n in range(0, self.number_of_neurons):
            if self.weights == False:
                neuron_weight = False
                specific_neuron_thresh = False
            else:
                neuron_weight = self.weights[n]
                specific_neuron_thresh = self.neuron_threshold[n]
            #neuron = SigmoidNeuron(neuron_weight) # here we are creating a neuron
            neuron = IntergrateFireNeuron(neuron_weight, specific_neuron_thresh) # here for the threshold we are accessing the specific neuron we want #the difference here is that IF needs a threshold because it sums the inputs 
            
            if self.weights != False:
                self.layer_activation.append(neuron.activity_level())
            
            self.neurons.append(neuron) #this is sending each neuron into the self.neuron vector
    

#Layer1 = Layer(3) # saying there are the number of neurons in this layer (can chage) 
#Layer1.create_neurons()
#print(Layer1.neurons)        
            
        
    

class Network:
    def __init__(self, number_of_layer, neurons_per_layer, weights, initial_activity, neuron_threshold):
        self.number_of_layer = number_of_layer
        self.layers = [] #this will be changing inside. It will be created on itsown. ^^^^^
        self.neurons_per_layer = neurons_per_layer
        self.weights = weights
        self.activations = [initial_activity]
        self.neuron_threshold = neuron_threshold
        
        
        
    def create_layers(self):
        
        
        for i in range(0, self.number_of_layer): #creating this for loop so we can iterate through the number of layers given from the outsiude
            number_of_neurons = self.neurons_per_layer[i] # this gives the number of neurons in that index example : index = [1,2] i==> 0=1 1 =2 
            weights_of_layer = False # needs to be false because we don't want to calculate the first layer of weights 
            
            if i!=0 : #but if we aren't working within the first layer then we want to caluclate the weights 
                weights_of_layer = []
                for j in range(0,len(self.weights)):
                    result_vector = [a * b for a, b in zip(self.activations[i-1], self.weights[j])]
                    weights_of_layer.append(result_vector)

            
            layer = Layer(number_of_neurons, weights_of_layer, neuron_threshold) #this is creating the object layer that takes the number of neurons 
            layer.create_neurons()
            
            if i!= 0:
                self.activations.append(layer.layer_activation)
                
            self.layers.append(layer) # putting every layer that we just created back into self.layers above ^^^^^
        
        print('This is our activation matrix of the whole Network')
        print(self.activations)
        return self.activations
        #return self.layers #return is the end of the funciton once all layers are created, it will give you all of the layers

    def winner_takes_all(self):
        #step 1 = get index of the last row - last (always equal to -->) index =  len(object)-1     
        last_row_index = len(self.activations) - 1
        
        #step 2 = get the last row object[index]
        last_row = self.activations[last_row_index]
        
        
        #step 3 = get the highest value in last row 
        highest_value = max(last_row)
        return highest_value 
        
        
        
        
#initialize the weights on the outside so they can change 
weights = [[[4,6],[5,8,1],[4,7,3]],[[3,7],[7,3],[4,2]]]# so we are going from number of layers (2), to number of neurons per layer. we have given eahc neuron a vector weights which will be summed. 

weights_matrix = [[2, 3, 5],[3,1,5],[3,6,4]]

initial_activity = [0.3, 0.6, 0.5]  #THIS IS INitializing the activity of the neurons in the first layer

neurons_per_layer = [3,3] #here we have 5 neurons in total in the network, the activity of the neurons in the second layer is changing depending on the activity of the first layer and this is how we are connecting the layers 


neuron_threshold = [5, 8, 3] #giving the threshold for the second layer of neurons 

Network1 = Network(2, neurons_per_layer, weights_matrix, initial_activity, neuron_threshold) #we've initialized two layers in the network 
y = Network1.create_layers()
print('This is the winner take all value in last layer')
print(Network1.winner_takes_all())




# we want to connect the layers. We will have to multiply the weight of each incoming neuron along with its activation weight. 
# We know the activation of the neurons in the first layer but will need to compute the others in the coming layers. 





        
        
        
        
        