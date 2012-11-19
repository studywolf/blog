import nef
import random

# constants / parameter setup etc
N = 50 # number of neurons
D = 3 # number of dimensions

def make_abs_val(dimensions, neurons, name, intercept=[0]):
    def mult_neg_one(x):
        return x[0] * -1 

    abs_val = nef.Network(name)

    input = abs_val.make('input', neurons=1, dimensions=dimensions, mode='direct') # create input relay
    output = abs_val.make('output', neurons=1, dimensions=dimensions, mode='direct') # create output relay
    
    for d in range(D): # create a positive and negative population for each dimension in the input signal
        abs_pos = abs_val.make('abs_pos%d'%d, neurons=neurons, dimensions=1, encoders=[[1]], intercept=intercept)
        abs_neg = abs_val.make('abs_neg%d'%d, neurons=neurons, dimensions=1, encoders=[[-1]], intercept=intercept)

        trans = [0 for i in range(D)]; trans[d] = 1
        abs_val.connect(input, abs_pos, transform=trans)
        abs_val.connect(input, abs_neg, transform=trans)
    
        abs_val.connect(abs_pos, output, transform=trans)
        abs_val.connect(abs_neg, output, transform=trans, func=mult_neg_one)

    return abs_val.network

net = nef.Network('network')

# Create absolute value subnetwork and add it to net
abs_val = make_abs_val(dimensions=D, neurons=N, name='abs_val')
net.add(abs_val) 

# Create function input
net.make_input('input', values=[random.random() for d in range(D)])

# Connect things up
net.connect('input', 'abs_val.input')

# Add it all to the Nengo world
net.add_to_nengo()
