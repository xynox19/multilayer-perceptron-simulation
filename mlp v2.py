import numpy as np
from sklearn.model_selection import train_test_split

def safe_int_input(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a valid integer.")

def safe_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid float.")

def get_parameters(): 
    inputs = safe_int_input("Input neurons: ")
    hidden_units = safe_int_input("Hidden neurons: ")
    outputs = safe_int_input("Output neurons: ")
    epochs = safe_int_input("Epochs: ")
    learning_rate = safe_float_input("Learning rate: ")
    activation = lambda x: 1 / (1 + np.exp(-x))
    derivative = lambda x: x * (1 - x)
    return inputs, hidden_units, outputs, epochs, activation, derivative, learning_rate

def load_data():
    choice = input("1. Manual entry\n2. Load from .txt file\nChoose: ")
    if choice == '1':
        data = eval(input("Enter data as [[x1,x2,...,y], [...]]: "))
    else:
        file_path = input("Path to .txt file: ")
        with open(file_path, 'r') as file:
            data = [list(map(float, line.strip().split())) for line in file if line.strip()]
    return np.array(data)

def normalize_data(data):
    features = data[:, :-1]
    labels = data[:, -1:]
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    return np.hstack((features, labels))

def show_structure(inputs, hidden_units, outputs):
    print(f"Structure: {inputs} input, {hidden_units} hidden, {outputs} output")

def initialize_weights(inputs, hidden_units, outputs):
    w1 = np.random.rand(inputs, hidden_units) - 0.5
    w2 = np.random.rand(hidden_units, outputs) - 0.5
    return w1, w2

def forward_pass(x, w1, w2, activation):
    h_input = np.dot(x, w1)
    h_output = activation(h_input)
    o_input = np.dot(h_output, w2)
    o_output = activation(o_input)
    return h_output, o_output

def backpropagation(x, y, w1, w2, activation, derivative, lr):
    h_out, y_pred = forward_pass(x, w1, w2, activation)
    error = y - y_pred
    delta_out = error * derivative(y_pred)
    delta_hidden = np.dot(delta_out, w2.T) * derivative(h_out)
    w2 += lr * np.dot(h_out.T, delta_out)
    w1 += lr * np.dot(x.T, delta_hidden)
    return w1, w2, np.mean(error ** 2)

def print_weights(w1, w2):
    print("Weights Input-Hidden:\n", w1)
    print("Weights Hidden-Output:\n", w2)

def main():
    params = get_parameters()
    data = load_data()
    data = normalize_data(data)
    inputs, hidden_units, outputs, epochs, activation, derivative, lr = params
    show_structure(inputs, hidden_units, outputs)
    w1, w2 = initialize_weights(inputs, hidden_units, outputs)

    for epoch in range(epochs):
        np.random.shuffle(data)
        total_loss = 0
        for sample in data:
            x = sample[:-1].reshape(1, -1)
            y = np.array(sample[-1]).reshape(1, -1)
            w1, w2, loss = backpropagation(x, y, w1, w2, activation, derivative, lr)
            total_loss += loss
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(data):.4f}")

    print_weights(w1, w2)

if __name__ == "__main__":
    main()
