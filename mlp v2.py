# Saanvi Sethi
# 13656663

# MULTILAYER PERCEPTRON MODELs

# Imports
import numpy as np

def Parameters(): 
    # Number of inputs
    inputs = int(input("Number of input neurons: "))
    # Hidden units
    hidden_units = int(input("Number of hidden neurons: "))
    # Output units
    outputs = int(input("Number of output neurons: "))
    # Epochs
    epochs = int(input("Number of epochs: "))
    # Learning rate
    learning_rate = float(input("Learning rate: "))
    # Activation function
    activation_func = lambda x: 1 / (1 + np.exp(-x))  # Sigmoid
    activation_derivative = lambda x: x * (1 - x)    # Derivative of Sigmoid
    
    return inputs, hidden_units, outputs, epochs, activation_func, activation_derivative, learning_rate


def Input():
    """
    Import training data or input own data.
    """
    data = np.array(eval(input("Enter training data as a list of lists (e.g., [[input1, input2, ..., target], [...]]): ")))
    return data


def Structure(inputs, hidden_units, outputs):
    """
    Organize the structure of the MLP.
    """
    print(f"Structure: {inputs} input neurons, {hidden_units} hidden neurons, {outputs} output neurons")


def Initialise(inputs, hidden_units, outputs):
    """
    Initialize weights between neurons for a fully connected graph with random weights.
    """
    weights_input_hidden = np.random.rand(inputs, hidden_units) - 0.5  # Input to hidden weights
    weights_hidden_output = np.random.rand(hidden_units, outputs) - 0.5  # Hidden to output weights
    return weights_input_hidden, weights_hidden_output


def Introduced(data, epochs):
    """
    Shuffle the dataset and introduce one training sample at a time for the specified number of epochs.
    """
    for epoch in range(epochs):
        np.random.shuffle(data)  # Random order
        for sample in data:
            yield sample


def Output(inputs, weights_input_hidden, weights_hidden_output, activation_func):
    """
    Calculate the output for hidden units and overall network output.
    """
    # Calculate hidden layer input and output
    hidden_input = np.dot(inputs, weights_input_hidden)  # Weighted sum for hidden layer
    hidden_output = activation_func(hidden_input)        # Apply Sigmoid activation function
    
    # Calculate final output layer input and output
    final_input = np.dot(hidden_output, weights_hidden_output)  # Weighted sum for output layer
    final_output = activation_func(final_input)                 # Apply Sigmoid activation function

    return hidden_output, final_output


def WeightUpdate(sample, weights_input_hidden, weights_hidden_output, activation_func, activation_derivative, learning_rate):
    """
    Update weights based on the difference between predicted and target outputs.
    """
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

    return weights_input_hidden, weights_hidden_output


def Print(weights_input_hidden, weights_hidden_output):
    """
    Print final neuron weights after training.
    """
    print("\nFinal Weights:")
    print("Weights between input and hidden layer:\n", weights_input_hidden)
    print("Weights between hidden and output layer:\n", weights_hidden_output)


def main():
    """
    Main function to execute the MLP model.
    """
    # Initialize parameters
    inputs, hidden_units, outputs, epochs, activation_func, activation_derivative, learning_rate = Parameters()
    data = Input()
    Structure(inputs, hidden_units, outputs)
    weights_input_hidden, weights_hidden_output = Initialise(inputs, hidden_units, outputs)

    # Training process
    for sample in Introduced(data, epochs):
        weights_input_hidden, weights_hidden_output = WeightUpdate(
            sample, weights_input_hidden, weights_hidden_output, activation_func, activation_derivative, learning_rate
        )

    # Print final weights
    Print(weights_input_hidden, weights_hidden_output)


if __name__ == "__main__":
    main()
