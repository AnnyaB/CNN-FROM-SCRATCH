import numpy as np  # Importing NumPy for array operations and mathematical calculations


# Sigmoid Activation Function
def sigmoid(x):
    """
    Sigmoid activation outputs values between 0 and 1, useful as probabilities.
    In probability terms, this models a "confidence level" that interprets any input as
    a likelihood score.

    Using the sigmoid function to map inputs to an output probability range [0,1].
    """
    return 1 / (1 + np.exp(-x))


# Derivative of Sigmoid for Backpropagation
def sigmoid_derivative(x):
    """
    Calculates the gradient of the Sigmoid function, which is essential for backpropagation.
    - Formula: sigmoid(x) * (1 - sigmoid(x))
    - This sensitivity helps adjust weights probabilistically during backpropagation.
    """
    return x * (1 - x)  # Gradient of sigmoid, used to update weights during backpropagation


# ReLU Activation Function
def relu(x):
    """
    ReLU (Rectified Linear Unit) treats only positive inputs as significant.
    Statistically, this is similar to "censoring" non-positive values.
    ReLU helps introduce non-linearity into the network.
    """
    return np.maximum(0, x)


# Derivative of ReLU for Backpropagation
def relu_derivative(x):
    """
    Derivative of ReLU function for backpropagation.
    - If the input is positive, the derivative is 1; otherwise, it’s 0.
    - This allows weight updates only for values that passed the censoring threshold.
    """
    return np.where(x > 0, 1, 0)


# Enhanced Neural Network Class with probabilistic and statistical concepts
class ProbabilisticNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize neural network with input, hidden, and output layers using probabilistic approaches.

        - Probabilistic Weight Initialization:
          Weights are now initialized using a normal distribution with a small standard deviation.
          This Bayesian-inspired approach assumes weights start as small random values rather than
          potentially large, which leads to more stable training.

        - Biases start at zero, allowing flexibility to shift activations during training.

        Parameters:
        - input_size: Number of input neurons (features)
        - hidden_size: Number of neurons in the hidden layer (arbitrary choice)
        - output_size: Number of neurons in the output layer (1 for binary classification)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Probabilistic weight initialization with smaller variance for stability
        self.weights_input_hidden = np.random.normal(0, 0.1, (input_size, hidden_size))
        self.bias_hidden = np.zeros((1, hidden_size))  # Initializing biases for the hidden layer

        self.weights_hidden_output = np.random.normal(0, 0.1, (hidden_size, output_size))
        self.bias_output = np.zeros((1, output_size))  # Initializing biases for the output layer

    def forward(self, X):
        """
        Forward pass computes the model's output with probabilistic activations.

        - Each layer's output can be interpreted as a confidence score.
        - The final Sigmoid output gives a probability, indicating model confidence in the binary classification.

        The forward pass can be broken down into:
        - Linear transformation (using matrix multiplication)
        - Non-linear transformation (via activation functions like ReLU and Sigmoid)
        """
        # Input to hidden layer (Matrix multiplication + Bias)
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = relu(self.hidden_input)  # Passing through ReLU activation for non-linearity

        # Hidden to output layer (Matrix multiplication + Bias)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)  # Final output is a probability

        return self.final_output

    def compute_loss(self, y_true, y_pred):
        """
        Cross-Entropy Loss for Probability Calibration:
        - Cross-entropy loss measures how well the model's probabilities align with the true labels.
        - Unlike mean squared error, cross-entropy directly evaluates the confidence of probabilistic
          outputs, giving a more accurate error signal in classification tasks.

        Formula: -y_true*log(y_pred) - (1 - y_true)*log(1 - y_pred)
        This is the application of calculus to calculate the error gradient.
        """
        epsilon = 1e-10  # Small constant to avoid log(0) and numerical instability
        loss = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        return loss

    def backward(self, X, y, learning_rate):
        """
        Backpropagation with Probabilistic Interpretation:

        - Cross-entropy gradients provide more robust adjustments for probability calibration,
          making the training process more aligned with minimizing misclassification probabilities.
        - Each adjustment step reflects a statistical likelihood, where the model gradually learns
          to improve confidence in correct predictions over incorrect ones.

        The chain rule and the matrix differentiation are used here to update weights.
        """
        # Error between actual and predicted probabilities (Gradient of loss with respect to output)
        error = y - self.final_output
        d_output = error * sigmoid_derivative(
            self.final_output)  # Applying Sigmoid gradient for probabilistic backpropagation

        # Backpropagating to hidden layer (Matrix multiplication + Element-wise multiplication for the derivative)
        d_hidden = d_output.dot(self.weights_hidden_output.T) * relu_derivative(self.hidden_output)

        # Update weights and biases using Gradient Descent (based on Linear Algebra and Calculus)
        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

        self.weights_input_hidden += X.T.dot(d_hidden) * learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        """
        Training with Cross-Entropy Loss for Probabilistic Calibration:

        - In each epoch, the model adjusts weights to increase the probability of correct predictions.
        - Cross-entropy loss serves as a probabilistic error signal, guiding each update to
          improve confidence in correct predictions.
        """
        for epoch in range(epochs):
            # Forward pass to compute probabilities
            predictions = self.forward(X)

            # Compute loss as a measure of fit
            loss = self.compute_loss(y, predictions)

            # Backward pass to update weights and biases
            self.backward(X, y, learning_rate)

            # Print loss every 1000 epochs to monitor training progress
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')


# Define inputs and outputs for an AND gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [0],
    [0],
    [1]
])

# Instantiate the enhanced neural network model
nn = ProbabilisticNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Train the model with probabilistic adjustments
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Making predictions
predictions = nn.forward(X)
print("\nPredictions after training:")
print(predictions)
