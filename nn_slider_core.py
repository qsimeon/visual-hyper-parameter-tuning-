"""Core neural network module for visual hyperparameter tuning.

This module provides a simple, fixed-architecture neural network implementation
with explicit weight matrices and biases. It supports parameter vectorization,
forward passes, and utilities for interactive weight tuning.

The network is designed for educational purposes and visual exploration of how
individual weights affect network outputs.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable


class SimpleMLPNetwork:
    """A simple multi-layer perceptron with explicit weight access.
    
    This network uses a fixed architecture (configurable layer sizes) and
    provides methods to:
    - Initialize weights and biases
    - Perform forward passes
    - Vectorize/unvectorize parameters for slider manipulation
    - Compute loss against targets
    
    Attributes:
        layer_sizes: List of integers defining network architecture
        weights: List of weight matrices for each layer
        biases: List of bias vectors for each layer
        activation: Activation function name ('relu', 'sigmoid', 'tanh')
    """
    
    def __init__(self, layer_sizes: List[int], activation: str = 'relu', seed: Optional[int] = 42):
        """Initialize the neural network.
        
        Args:
            layer_sizes: List of layer sizes, e.g., [2, 4, 3, 1] for
                        2 inputs, two hidden layers (4 and 3 units), 1 output
            activation: Activation function ('relu', 'sigmoid', 'tanh')
            seed: Random seed for reproducibility
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # Xavier/Glorot initialization
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)
    
    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply the activation function.
        
        Args:
            x: Input array
            
        Returns:
            Activated array
        """
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            return x
    
    def forward(self, x: np.ndarray, return_activations: bool = False) -> np.ndarray | Tuple[np.ndarray, List[np.ndarray]]:
        """Perform forward pass through the network.
        
        Args:
            x: Input array of shape (batch_size, input_dim) or (input_dim,)
            return_activations: If True, return intermediate activations
            
        Returns:
            Output array, or (output, activations) if return_activations=True
        """
        # Handle single sample
        single_sample = False
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_sample = True
        
        activations = [x]
        current = x
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            current = current @ w + b
            # Apply activation to all layers except possibly the last
            if i < len(self.weights) - 1:
                current = self._apply_activation(current)
            activations.append(current)
        
        output = current
        if single_sample:
            output = output.flatten()
        
        if return_activations:
            return output, activations
        return output
    
    def vectorize_parameters(self) -> np.ndarray:
        """Convert all weights and biases to a single parameter vector.
        
        Returns:
            1D numpy array containing all parameters
        """
        params = []
        for w, b in zip(self.weights, self.biases):
            params.append(w.flatten())
            params.append(b.flatten())
        return np.concatenate(params)
    
    def unvectorize_parameters(self, param_vector: np.ndarray) -> None:
        """Update network weights and biases from a parameter vector.
        
        Args:
            param_vector: 1D array of parameters
        """
        idx = 0
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            b_size = self.biases[i].size
            
            self.weights[i] = param_vector[idx:idx + w_size].reshape(self.weights[i].shape)
            idx += w_size
            
            self.biases[i] = param_vector[idx:idx + b_size].reshape(self.biases[i].shape)
            idx += b_size
    
    def get_parameter_info(self) -> List[Dict[str, any]]:
        """Get information about each parameter for slider creation.
        
        Returns:
            List of dictionaries with parameter metadata
        """
        param_info = []
        param_idx = 0
        
        for layer_idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            # Weight parameters
            for i in range(w.shape[0]):
                for j in range(w.shape[1]):
                    param_info.append({
                        'index': param_idx,
                        'name': f'W{layer_idx}[{i},{j}]',
                        'layer': layer_idx,
                        'type': 'weight',
                        'position': (i, j),
                        'value': w[i, j]
                    })
                    param_idx += 1
            
            # Bias parameters
            for i in range(b.shape[0]):
                param_info.append({
                    'index': param_idx,
                    'name': f'b{layer_idx}[{i}]',
                    'layer': layer_idx,
                    'type': 'bias',
                    'position': i,
                    'value': b[i]
                })
                param_idx += 1
        
        return param_info
    
    def compute_loss(self, x: np.ndarray, y_true: np.ndarray, loss_type: str = 'mse') -> float:
        """Compute loss between predictions and targets.
        
        Args:
            x: Input data
            y_true: Target outputs
            loss_type: Type of loss ('mse', 'mae')
            
        Returns:
            Loss value
        """
        y_pred = self.forward(x)
        
        if loss_type == 'mse':
            return np.mean((y_pred - y_true) ** 2)
        elif loss_type == 'mae':
            return np.mean(np.abs(y_pred - y_true))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters in the network.
        
        Returns:
            Total parameter count
        """
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases))


def create_demo_dataset(task: str = 'xor', n_samples: int = 100, noise: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Create a synthetic dataset for demonstration.
    
    Args:
        task: Task type ('xor', 'circle', 'linear', 'sine')
        n_samples: Number of samples to generate
        noise: Noise level to add to outputs
        
    Returns:
        Tuple of (X, y) where X is inputs and y is targets
    """
    if task == 'xor':
        # XOR problem
        X = np.random.rand(n_samples, 2) * 2 - 1
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float).reshape(-1, 1)
        y += np.random.randn(n_samples, 1) * noise
        
    elif task == 'circle':
        # Circle classification
        X = np.random.rand(n_samples, 2) * 2 - 1
        y = (np.sum(X ** 2, axis=1) < 0.5).astype(float).reshape(-1, 1)
        y += np.random.randn(n_samples, 1) * noise
        
    elif task == 'linear':
        # Simple linear relationship
        X = np.random.rand(n_samples, 2) * 2 - 1
        y = (X[:, 0] + X[:, 1]).reshape(-1, 1)
        y += np.random.randn(n_samples, 1) * noise
        
    elif task == 'sine':
        # Sine wave
        X = np.random.rand(n_samples, 1) * 4 * np.pi - 2 * np.pi
        y = np.sin(X)
        y += np.random.randn(n_samples, 1) * noise
        
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return X, y


def compute_activation_statistics(activations: List[np.ndarray]) -> Dict[str, List[float]]:
    """Compute statistics about layer activations.
    
    Useful for understanding network behavior and identifying dead neurons.
    
    Args:
        activations: List of activation arrays from forward pass
        
    Returns:
        Dictionary with statistics for each layer
    """
    stats = {
        'mean': [],
        'std': [],
        'sparsity': [],  # Fraction of zeros (for ReLU)
        'max': [],
        'min': []
    }
    
    for act in activations[1:]:  # Skip input layer
        stats['mean'].append(float(np.mean(act)))
        stats['std'].append(float(np.std(act)))
        stats['sparsity'].append(float(np.mean(act == 0)))
        stats['max'].append(float(np.max(act)))
        stats['min'].append(float(np.min(act)))
    
    return stats
