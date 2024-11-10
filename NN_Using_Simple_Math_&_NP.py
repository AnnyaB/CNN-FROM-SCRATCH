import numpy as np  # Importing NumPy, a library that helps with array manipulations and mathematical operations.


# Define the Sigmoid Activation Function
def sigmoid(x):
    """
    Sigmoid function squashes the input between 0 and 1.
    It is useful for binary classification and probability outputs.

    This function is important because it introduces non-linearity to the model.
    Without non-linearity, the model would essentially be a linear model, which
    can't learn complex patterns.
    """
    return 1 / (1 + np.exp(-x))  # Sigmoid formula: 1 / (1 + e^(-x))


# Define the Derivative of the Sigmoid Function
def sigmoid_derivative(x):
    """
    Derivative of the Sigmoid function.

    The derivative of the sigmoid function is important for backpropagation.
    It helps the model adjust the weights during training by telling how much change
    in the output results from a change in the input.
    """
    return x * (1 - x)  # Sigmoid derivative formula


# Define the ReLU Activation Function (used in the hidden layer)
def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function.

    ReLU returns the input if it's greater than 0, otherwise, it returns 0.
    It is popular because it helps with faster training by avoiding problems with
    gradients vanishing during training.
    """
    return np.maximum(0, x)  # Returns the input if it's positive, otherwise returns 0.


# Define the Derivative of the ReLU Function
def relu_derivative(x):
    """
    Derivative of the ReLU function.

    This derivative is important for adjusting the weights during backpropagation.
    ReLU's derivative is 1 for positive values and 0 for non-positive values.
    """
    return np.where(x > 0, 1, 0)  # ReLU derivative: 1 if input > 0, else 0.


# Neural Network Class Definition
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the neural network with sizes for input, hidden, and output layers.

        We also randomly initialize the weights and biases here. Weights are the
        parameters that the model learns, and biases help shift the activation function.
        """
        self.input_size = input_size  # Number of input features
        self.hidden_size = hidden_size  # Number of neurons in the hidden layer
        self.output_size = output_size  # Number of output neurons

        # Randomly initialize the weights between input and hidden layers, and between hidden and output layers.
        # Weights are initialized randomly to break symmetry (they can't all be the same value).
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)  # Weights for input to hidden layer
        self.bias_hidden = np.zeros((1, hidden_size))  # Bias for hidden layer (initialized to 0)

        self.weights_hidden_output = np.random.randn(hidden_size, output_size)  # Weights for hidden to output layer
        self.bias_output = np.zeros((1, output_size))  # Bias for output layer (initialized to 0)

    def forward(self, X):
        """
        Perform forward propagation.

        In forward propagation, the input data flows through the network, and
        activations are computed for each layer. The result is the output of the network.

        X is the input data, where each row is an example.
        """
        # Compute the input to the hidden layer (weighted sum of inputs + bias)
        self.hidden_input = np.dot(X,
                                   self.weights_input_hidden) + self.bias_hidden  # Matrix multiplication for hidden layer
        self.hidden_output = relu(self.hidden_input)  # Apply ReLU activation to the hidden layer's input

        # Compute the input to the output layer (weighted sum of hidden layer's output + bias)
        self.final_input = np.dot(self.hidden_output,
                                  self.weights_hidden_output) + self.bias_output  # Matrix multiplication for output layer
        self.final_output = sigmoid(self.final_input)  # Apply Sigmoid activation to the output layer's input

        return self.final_output  # Return the output of the network

    def backward(self, X, y, learning_rate):
        """
        Perform backpropagation to update weights and biases.

        Backpropagation adjusts the model's parameters (weights and biases) by calculating
        the gradient of the loss function with respect to each parameter, and then updating
        the parameters to minimize the loss.

        X is the input data, y is the true labels, and learning_rate is a factor that controls
        how big the weight updates are.
        """
        # Calculate the error (difference between the predicted and actual values)
        error = y - self.final_output  # How much the prediction was wrong

        # Calculate the gradient (derivative) of the error with respect to the output layer
        # This tells us how much the output layer needs to change to reduce the error.
        d_output = error * sigmoid_derivative(self.final_output)  # Apply sigmoid derivative to the output layer

        # Calculate the gradient of the error with respect to the hidden layer
        # This tells us how much the hidden layer needs to change to reduce the error in the output.
        d_hidden = d_output.dot(self.weights_hidden_output.T) * relu_derivative(
            self.hidden_output)  # Apply ReLU derivative

        # Update the weights and biases using the gradients calculated above
        self.weights_hidden_output += self.hidden_output.T.dot(
            d_output) * learning_rate  # Update weights for hidden-output layer
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate  # Update biases for output layer

        self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate  # Update weights for input-hidden layer
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate  # Update biases for hidden layer

    def train(self, X, y, epochs, learning_rate):
        """
        Train the neural network using forward and backward propagation.

        In training, the model learns by repeatedly updating its weights and biases to reduce
        the error on the training data. This is done through multiple iterations (epochs).

        Each epoch is a pass through the entire training data.
        """
        for epoch in range(epochs):
            self.forward(X)  # Perform forward propagation to compute the model's output
            self.backward(X, y, learning_rate)  # Perform backpropagation to adjust weights and biases

            # Monitor and print the loss (error) every 1000 epochs to check the model's progress
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.final_output))  # Mean Squared Error (MSE) loss function
                print(f'Epoch {epoch}, Loss: {loss}')  # Print loss at regular intervals


# Create the training data (X) and labels (y) for the AND gate
X = np.array([
    [0, 0],  # Input: 0 AND 0
    [0, 1],  # Input: 0 AND 1
    [1, 0],  # Input: 1 AND 0
    [1, 1]  # Input: 1 AND 1
])

y = np.array([
    [0],  # Output: 0 AND 0 = 0
    [0],  # Output: 0 AND 1 = 0
    [0],  # Output: 1 AND 0 = 0
    [1]  # Output: 1 AND 1 = 1
])

# Create an instance of the neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Train the network with the training data for 10,000 epochs and a learning rate of 0.1
nn.train(X, y, epochs=10000, learning_rate=0.1)

# After training, use the network to make predictions on the training data
predictions = nn.forward(X)

print("\nPredictions after training:")
print(predictions)

# Time Complexity Analysis:
# 1. Forward pass (for each epoch):
#    - Input to hidden layer (matrix multiplication): O(N * H) where N is the number of samples and H is the number of hidden neurons.
#    - Hidden to output layer (matrix multiplication): O(N * O) where O is the number of output neurons.
#    - Total for forward pass: O(N * (H + O)).

# 2. Backpropagation (for each epoch):
#    - Calculating gradients for output and hidden layers involves matrix multiplications and derivative applications: O(N * (H + O)).

# 3. Total Complexity per Epoch:
#    - Forward pass + Backpropagation: O(N * (H + O)) for each epoch.
#    - For E epochs, the total time complexity is: O(E * N * (H + O)).

# With a high number of epochs and larger networks, this can become computationally expensive.
