#Saanvi Sethi
#13656663

#MULTILAYER PERCEPTRON MODEL

#imports
import numpy as np

def Parameters(): #allows the user to enter parameters needed to create the network
    #number of inputs
    inputs=int(input("number of input neurons: "))
    #hidden units
    hidden_units=int(input("number of hidden neurons: "))
    #output units
    outputs=int(input("number of output neurons: "))
    #epochs
    epoch=int(input("number of epochs: "))
    #learning rate
    learn_rate=float(input("learn rate: "))
    return inputs, hidden_units, outputs, epochs, activate, learning_rate

    #activation function
    activation_func = lambda x: 1 / (1 + np.exp(-x))  # Sigmoid
    activation_derivative = lambda x: x * (1 - x)    # Derivative of Sigmoid
    
    return inputs, hidden_units, outputs, epochs, activation_func, activation_derivative, learning_rate

    # Learning rate
    learn_rate = float(input("Learning rate: "))
    return inputs, hidden_units, outputs, epochs, activate, activate_derivative, learn_rate

    
def Input():
    #import training data
    #input own data
    data = np.array(eval(input("training data as a list of lists: ")))  # e.g., [[input1, input2, ..., target], [...]]
    return data
    
def Structure(inputs, hidden_units, outputs): #organise structure of mlp
    print(f"Structure: {inputs}, {hidden_units}, {outputs}")    
    
def Initialise(inputs, hidden_units, outputs):
    #random weight between 0 and 1 produced for each connection between neurons
    #fully connected graph
    weights_input_hidden = np.random.rand(inputs, hidden_units) - 0.5  # Input to hidden weights
    weights_hidden_output = np.random.rand(hidden_units, outputs) - 0.5  # Hidden to output weights
    return weights_input_hidden, weights_hidden_output

    
def Introduced(data, epochs):
    #repeatedly introduce one training sample at a time
    #at the input layer.  The full training set should be introduced
    #in a random order for the
    #number of times represented by the number of epochs
    for epoch in range(epochs):
        np.random.shuffle(data)  # random order 
        for sample in data:
            yield sample
    
def Output(inputs, weights_input_hidden, weights_hidden_output, activation_func):
    #output calculationn hidden units
    hidden_input = np.dot(inputs, weights_input_hidden)  # Weighted sum for hidden layer
    hidden_output = activation_func(hidden_input)        # Apply Sigmoid activation function
    
    # Calculate final output layer input and overall network output
    final_input = np.dot(hidden_output, weights_hidden_output)  # Weighted sum for output layer
    final_output = activation_func(final_input)                 # Apply Sigmoid activation function

    return hidden_output, final_output
        
    
    
def WeightUpdate(weights, inputs, target_output, predicted_output, learning_rate):
    #difference between predicted output and target output
    #update weight
    global weights_input_hidden, weights_hidden_output, sample, learning_rate, activation_derivative
    
    # Separate input features and target output
    input_sample = sample[:-1].reshape(1, -1)  # Input features
    target_output = np.array(sample[-1]).reshape(1, -1)  # Target output

    # Forward pass
    hidden_output, predicted_output = Output(input_sample, weights_input_hidden, weights_hidden_output, activation_func)

    # Backpropagation
    # Update weights from hidden to output
    output_error = target_output - predicted_output  # Error at output
    delta_output = output_error * activation_derivative(predicted_output)
    weights_hidden_output += learning_rate * np.dot(hidden_output.T, delta_output)  # Update weights hidden -> output

    # Update weights from input to hidden
    hidden_error = np.dot(delta_output, weights_hidden_output.T)  # Error propagated back to hidden layer
    delta_hidden = hidden_error * activation_derivative(hidden_output)
    weights_input_hidden += learning_rate * np.dot(input_sample.T, delta_hidden)  # Update weights input -> hidden


    return hidden_output, final_output
    
def Print():
    #prints neuron weights
    print("\nFinal Weights:")
    print("input and hidden layer weights:\n", weights_input_hidden)
    print("hidden and output layer weights:\n", weights_hidden_output)

#running the mlp    
def main():
    global activation_func, activation_derivative, weights_input_hidden, weights_hidden_output, sample, learning_rate
    inputs, hidden_units, outputs, epochs, activation_func, activation_derivative, learning_rate = Parameters()
    data = Input()
    Structure(inputs, hidden_units, outputs)
    weights_input_hidden, weights_hidden_output = Initialise(inputs, hidden_units, outputs)

    #train mlp
    for sample in Introduced(data, epochs):
        WeightUpdate()

    # Print final weights from print FUNCTION
    Print()


if __name__ == "__main__":
    main()



#test data
    #learning rate =0.1
    #epochs = 20

    
