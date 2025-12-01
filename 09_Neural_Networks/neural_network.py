"""
Neural Networks Implementation
==============================

This module provides basic neural network implementations.

Author: Machine Learning Hands-on
Date: 2025-12-01
"""

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt


# Activation functions
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid."""
    return x * (1 - x)

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU."""
    return (x > 0).astype(float)

def softmax(x):
    """Softmax activation function."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class NeuralNetworkFromScratch:
    """
    Simple feedforward neural network implementation.
    
    Architecture: Input -> Hidden Layer -> Output Layer
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize neural network.
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of neurons in hidden layer
        output_size : int
            Number of output neurons
        learning_rate : float
            Learning rate
        """
        self.lr = learning_rate
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        self.loss_history = []
        
    def forward(self, X):
        """
        Forward propagation.
        
        Parameters:
        -----------
        X : array-like
            Input data
            
        Returns:
        --------
        output : array-like
            Network output
        cache : dict
            Intermediate values for backpropagation
        """
        # Hidden layer
        z1 = np.dot(X, self.W1) + self.b1
        a1 = sigmoid(z1)
        
        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = sigmoid(z2)
        
        cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        return a2, cache
    
    def backward(self, X, y, cache):
        """
        Backward propagation.
        
        Parameters:
        -----------
        X : array-like
            Input data
        y : array-like
            True labels
        cache : dict
            Cached values from forward pass
        """
        m = X.shape[0]
        a1, a2 = cache['a1'], cache['a2']
        
        # Output layer gradients
        dz2 = a2 - y
        dW2 = (1/m) * np.dot(a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dz1 = np.dot(dz2, self.W2.T) * sigmoid_derivative(a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def fit(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Target values
        epochs : int
            Number of training epochs
        verbose : bool
            Print training progress
        """
        for epoch in range(epochs):
            # Forward pass
            output, cache = self.forward(X)
            
            # Calculate loss
            loss = np.mean((output - y) ** 2)
            self.loss_history.append(loss)
            
            # Backward pass
            self.backward(X, y, cache)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return self
    
    def predict(self, X):
        """Predict output."""
        output, _ = self.forward(X)
        return (output > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        output, _ = self.forward(X)
        return output


def plot_loss_history(loss_history, title='Training Loss'):
    """Plot training loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_neural_network_architecture(input_size, hidden_sizes, output_size):
    """
    Visualize neural network architecture.
    
    Parameters:
    -----------
    input_size : int
        Number of input neurons
    hidden_sizes : list
        Number of neurons in each hidden layer
    output_size : int
        Number of output neurons
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layers = [input_size] + hidden_sizes + [output_size]
    layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(hidden_sizes))] + ['Output']
    
    # Calculate positions
    max_neurons = max(layers)
    layer_spacing = 1.0 / (len(layers) - 1)
    
    for i, (size, name) in enumerate(zip(layers, layer_names)):
        x = i * layer_spacing
        neuron_spacing = 1.0 / (size + 1)
        
        for j in range(size):
            y = (j + 1) * neuron_spacing
            circle = plt.Circle((x, y), 0.02, color='skyblue', ec='black', zorder=4)
            ax.add_patch(circle)
            
            # Draw connections to next layer
            if i < len(layers) - 1:
                next_size = layers[i + 1]
                next_spacing = 1.0 / (next_size + 1)
                for k in range(next_size):
                    next_y = (k + 1) * next_spacing
                    ax.plot([x, x + layer_spacing], [y, next_y], 
                           'gray', alpha=0.3, linewidth=0.5, zorder=1)
        
        # Add layer label
        ax.text(x, -0.1, f'{name}\n({size})', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 1.1)
    ax.axis('off')
    plt.title('Neural Network Architecture', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_iris
    
    # Generate dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_redundant=5, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Method 1: Using scikit-learn
    print("=" * 50)
    print("Scikit-learn Neural Network (MLP)")
    print("=" * 50)
    
    sklearn_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                                  max_iter=1000, random_state=42)
    sklearn_model.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test_scaled)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_sklearn):.4f}")
    print(f"Number of iterations: {sklearn_model.n_iter_}")
    print(f"Loss: {sklearn_model.loss_:.4f}")
    
    # Method 2: From scratch (simple binary classification)
    print("\n" + "=" * 50)
    print("Neural Network from Scratch")
    print("=" * 50)
    
    # Reshape y for binary classification
    y_train_reshaped = y_train.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    
    custom_model = NeuralNetworkFromScratch(
        input_size=X_train_scaled.shape[1],
        hidden_size=32,
        output_size=1,
        learning_rate=0.1
    )
    
    custom_model.fit(X_train_scaled, y_train_reshaped, epochs=1000, verbose=False)
    y_pred_custom = custom_model.predict(X_test_scaled)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_custom):.4f}")
    
    # Visualizations
    print("\n" + "=" * 50)
    print("Visualizations")
    print("=" * 50)
    
    plot_loss_history(custom_model.loss_history, 'Neural Network Training Loss')
    plot_neural_network_architecture(20, [64, 32], 1)
    
    # Plot sklearn loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(sklearn_model.loss_curve_)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Scikit-learn MLP Training Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
