import nef
import random

# constants / parameter setup etc
N = 50 # number of neurons
D = 3 # number of dimensions

def make_abs_val(name, neurons, dimensions, intercept=[0]):
    def mult_neg_one(x):
        return x[0] * -1 

    abs_val = nef.Network(name)

    abs_val.make('input', neurons=1, dimensions=dimensions, mode='direct') # create input relay
    abs_val.make('output', neurons=1, dimensions=dimensions, mode='direct') # create output relay
    
    for d in range(dimensions): # create a positive and negative population for each dimension in the input signal
        abs_val.make('abs_pos%d'%d, neurons=neurons, dimensions=1, encoders=[[1]], intercept=intercept)
        abs_val.make('abs_neg%d'%d, neurons=neurons, dimensions=1, encoders=[[-1]], intercept=intercept)

        abs_val.connect('input', 'abs_pos%d'%d, index_pre=d)
        abs_val.connect('input', 'abs_neg%d'%d, index_pre=d)
    
        abs_val.connect('abs_pos%d'%d, 'output', index_post=d)
        abs_val.connect('abs_neg%d'%d, 'output', index_post=d, func=mult_neg_one)

    return abs_val.network

net = nef.Network('network')

# Create absolute value subnetwork and add it to net
net.add(make_abs_val(name='abs_val', dimensions=D, neurons=N))

# Create function input
net.make_input('input', values=[random.random() for d in range(D)])

# Connect things up
net.connect('input', 'abs_val.input')

# Add it all to the Nengo world
net.add_to_nengo()
