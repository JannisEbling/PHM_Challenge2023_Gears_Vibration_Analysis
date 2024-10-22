import torch
import torch.nn as nn


class CustomNonFullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_size, connectivity_matrix):
        super(CustomNonFullyConnectedLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Create a parameter to represent the connectivity matrix
        self.connectivity_matrix = nn.Parameter(
            torch.Tensor(connectivity_matrix), requires_grad=False
        )

        # Define the weights and biases for the layer
        self.weights = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.biases = nn.Parameter(torch.Tensor(hidden_size))

        # Initialize the weights and biases
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.biases)

    def forward(self, x):
        # Perform the computation for the layer
        # Apply the connectivity pattern using element-wise multiplication with the connectivity matrix
        weights_filtered = self.weights * self.connectivity_matrix
        output = torch.matmul(x, weights_filtered.t()) + self.biases

        return output


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, connectivity_matrix):
        super(NeuralNetwork, self).__init__()

        self.custom_layer = CustomNonFullyConnectedLayer(
            input_size, hidden_size, connectivity_matrix
        )

        self.fc1 = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x = self.custom_layer(x)
        x = self.fc1(x)

        return x
