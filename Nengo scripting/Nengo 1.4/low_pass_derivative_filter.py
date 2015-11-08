import nef

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

def make_dlowpass(name, neurons, dimensions, radius=10, tau_inhib=0.005, inhib_scale=10):

    dlowpass = nef.Network(name)

    dlowpass.make('input', neurons=1, dimensions=dimensions, mode='direct') # create input relay
    output = dlowpass.make('output', neurons=dimensions*neurons, dimensions=dimensions) # create output relay
    
    # now we track the derivative of sum, and only let output relay the input
    # if the derivative is below a given threshold
    dlowpass.make('derivative', neurons=radius*neurons, dimensions=2, radius=radius) # create population to calculate the derivative
    dlowpass.connect('derivative', 'derivative', index_pre=0, index_post=1, pstc=0.1) # set up recurrent connection
    dlowpass.add(make_abs_val(name='abs_val', neurons=neurons, dimensions=1, intercept=(.2,1))) # create a subnetwork to calculate the absolute value  

    # connect it up!
    dlowpass.connect('input', 'output') # set up communication channel
    dlowpass.connect('input', 'derivative', index_post=0)
    
    def sub(x):
        return [x[0] - x[1]]
    dlowpass.connect('derivative', 'abs_val.input', func=sub)
        
    # set up inhibitory matrix
    inhib_matrix = [[-inhib_scale]] * neurons * dimensions
    output.addTermination('inhibition', inhib_matrix, tau_inhib, False)
    dlowpass.connect('abs_val.output', output.getTermination('inhibition'))

    return dlowpass.network

# Create network
net = nef.Network('net')

# Create / add low pass derivative filter
net.add(make_dlowpass(name='dlowpass', neurons=N, dimensions=D))

# Make function input
net.make_input('input_function', values=[0]*D)

# Connect up function input to filter
net.connect('input_function', 'dlowpass.input')

# Add it all to Nengo
net.add_to_nengo()

