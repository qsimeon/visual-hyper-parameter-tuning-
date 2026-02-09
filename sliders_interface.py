"""Interactive slider interface for neural network weight tuning.

This module provides an ipywidgets-based interface for visually tuning
individual neural network weights and biases. It creates sliders for each
parameter and displays live updates of network outputs and metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable, Dict, List, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output
import io
import base64
from matplotlib.figure import Figure

try:
    from nn_slider_core import SimpleMLPNetwork, compute_activation_statistics
except ImportError:
    # For standalone testing
    pass


class NetworkSliderInterface:
    """Interactive slider interface for neural network weight tuning.
    
    This class creates an ipywidgets interface with:
    - A slider for each weight and bias parameter
    - Live visualization of network outputs
    - Metrics display (loss, activation statistics)
    - Optional decision boundary or function visualization
    
    Attributes:
        network: The neural network to tune
        X_data: Input data for evaluation
        y_data: Target outputs
        sliders: Dictionary of parameter sliders
        output_widget: Widget for displaying plots
    """
    
    def __init__(self, 
                 network: 'SimpleMLPNetwork',
                 X_data: np.ndarray,
                 y_data: np.ndarray,
                 slider_range: Tuple[float, float] = (-5.0, 5.0),
                 slider_step: float = 0.01):
        """Initialize the slider interface.
        
        Args:
            network: Neural network instance to tune
            X_data: Input data for visualization and loss computation
            y_data: Target outputs
            slider_range: (min, max) range for sliders
            slider_step: Step size for slider values
        """
        self.network = network
        self.X_data = X_data
        self.y_data = y_data
        self.slider_range = slider_range
        self.slider_step = slider_step
        
        # Store initial parameters
        self.initial_params = network.vectorize_parameters().copy()
        
        # Create parameter info
        self.param_info = network.get_parameter_info()
        
        # Create sliders
        self.sliders: Dict[int, widgets.FloatSlider] = {}
        self._create_sliders()
        
        # Output widget for plots
        self.output_widget = widgets.Output()
        
        # Metrics display
        self.metrics_widget = widgets.HTML(value="")
        
        # Control buttons
        self.reset_button = widgets.Button(description="Reset to Initial")
        self.reset_button.on_click(self._reset_parameters)
        
        self.randomize_button = widgets.Button(description="Randomize")
        self.randomize_button.on_click(self._randomize_parameters)
        
        # Layout
        self.interface = None
        self._build_interface()
    
    def _create_sliders(self) -> None:
        """Create a slider for each parameter."""
        for param in self.param_info:
            idx = param['index']
            slider = widgets.FloatSlider(
                value=param['value'],
                min=self.slider_range[0],
                max=self.slider_range[1],
                step=self.slider_step,
                description=param['name'],
                continuous_update=False,  # Update only on release for performance
                readout=True,
                readout_format='.3f',
                layout=widgets.Layout(width='400px')
            )
            slider.observe(self._on_slider_change, names='value')
            self.sliders[idx] = slider
    
    def _on_slider_change(self, change) -> None:
        """Handle slider value changes."""
        # Update network parameters
        param_vector = np.array([self.sliders[i].value for i in range(len(self.sliders))])
        self.network.unvectorize_parameters(param_vector)
        
        # Update visualization
        self._update_visualization()
    
    def _reset_parameters(self, button) -> None:
        """Reset all parameters to initial values."""
        for idx, value in enumerate(self.initial_params):
            self.sliders[idx].value = value
    
    def _randomize_parameters(self, button) -> None:
        """Randomize all parameters."""
        for idx in range(len(self.sliders)):
            self.sliders[idx].value = np.random.randn() * 0.5
    
    def _build_interface(self) -> None:
        """Build the complete interface layout."""
        # Group sliders by layer
        slider_groups = {}
        for param in self.param_info:
            layer = param['layer']
            if layer not in slider_groups:
                slider_groups[layer] = []
            slider_groups[layer].append(self.sliders[param['index']])
        
        # Create accordion for sliders
        slider_boxes = []
        for layer in sorted(slider_groups.keys()):
            layer_box = widgets.VBox(slider_groups[layer])
            slider_boxes.append(layer_box)
        
        slider_accordion = widgets.Accordion(children=slider_boxes)
        for i in range(len(slider_boxes)):
            slider_accordion.set_title(i, f'Layer {i} Parameters')
        
        # Control panel
        control_panel = widgets.HBox([self.reset_button, self.randomize_button])
        
        # Left panel: sliders and controls
        left_panel = widgets.VBox([
            widgets.HTML("<h3>Network Parameters</h3>"),
            control_panel,
            slider_accordion
        ], layout=widgets.Layout(width='450px', overflow_y='auto', max_height='600px'))
        
        # Right panel: visualization and metrics
        right_panel = widgets.VBox([
            widgets.HTML("<h3>Network Output & Metrics</h3>"),
            self.metrics_widget,
            self.output_widget
        ], layout=widgets.Layout(width='600px'))
        
        # Complete interface
        self.interface = widgets.HBox([left_panel, right_panel])
        
        # Initial visualization
        self._update_visualization()
    
    def _update_visualization(self) -> None:
        """Update the output visualization and metrics."""
        with self.output_widget:
            clear_output(wait=True)
            
            # Compute predictions and loss
            y_pred = self.network.forward(self.X_data)
            loss = self.network.compute_loss(self.X_data, self.y_data)
            
            # Get activations for statistics
            _, activations = self.network.forward(self.X_data, return_activations=True)
            act_stats = compute_activation_statistics(activations)
            
            # Update metrics
            metrics_html = f"""
            <div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px;'>
                <b>Loss (MSE):</b> {loss:.6f}<br>
                <b>Mean Prediction:</b> {np.mean(y_pred):.4f}<br>
                <b>Std Prediction:</b> {np.std(y_pred):.4f}<br>
            </div>
            """
            self.metrics_widget.value = metrics_html
            
            # Create visualization
            fig = self._create_visualization_plot(y_pred, activations, act_stats)
            plt.show()
    
    def _create_visualization_plot(self, 
                                   y_pred: np.ndarray, 
                                   activations: List[np.ndarray],
                                   act_stats: Dict[str, List[float]]) -> Figure:
        """Create visualization plots.
        
        Args:
            y_pred: Network predictions
            activations: Layer activations
            act_stats: Activation statistics
            
        Returns:
            Matplotlib figure
        """
        # Determine plot layout based on input dimension
        input_dim = self.X_data.shape[1]
        
        if input_dim == 1:
            # 1D input: plot function
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Sort for plotting
            sort_idx = np.argsort(self.X_data[:, 0])
            axes[0].plot(self.X_data[sort_idx, 0], self.y_data[sort_idx], 'o', 
                        label='Target', alpha=0.6)
            axes[0].plot(self.X_data[sort_idx, 0], y_pred[sort_idx], '-', 
                        label='Prediction', linewidth=2)
            axes[0].set_xlabel('Input')
            axes[0].set_ylabel('Output')
            axes[0].set_title('Network Function')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
        elif input_dim == 2:
            # 2D input: scatter plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Target vs prediction scatter
            scatter = axes[0].scatter(self.X_data[:, 0], self.X_data[:, 1], 
                                     c=y_pred.flatten(), cmap='viridis', s=50, alpha=0.7)
            axes[0].set_xlabel('X1')
            axes[0].set_ylabel('X2')
            axes[0].set_title('Network Output (color)')
            plt.colorbar(scatter, ax=axes[0])
            axes[0].grid(True, alpha=0.3)
            
        else:
            # Higher dimensional: just show predictions vs targets
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].scatter(self.y_data, y_pred, alpha=0.5)
            axes[0].plot([self.y_data.min(), self.y_data.max()], 
                        [self.y_data.min(), self.y_data.max()], 'r--', linewidth=2)
            axes[0].set_xlabel('Target')
            axes[0].set_ylabel('Prediction')
            axes[0].set_title('Prediction vs Target')
            axes[0].grid(True, alpha=0.3)
        
        # Activation statistics plot
        if len(act_stats['mean']) > 0:
            layers = list(range(len(act_stats['mean'])))
            axes[1].plot(layers, act_stats['mean'], 'o-', label='Mean', linewidth=2)
            axes[1].plot(layers, act_stats['std'], 's-', label='Std', linewidth=2)
            axes[1].plot(layers, act_stats['sparsity'], '^-', label='Sparsity', linewidth=2)
            axes[1].set_xlabel('Layer')
            axes[1].set_ylabel('Value')
            axes[1].set_title('Activation Statistics')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xticks(layers)
        
        plt.tight_layout()
        return fig
    
    def display(self) -> None:
        """Display the interface."""
        display(self.interface)
    
    def get_current_parameters(self) -> np.ndarray:
        """Get current parameter values from sliders.
        
        Returns:
            Current parameter vector
        """
        return np.array([self.sliders[i].value for i in range(len(self.sliders))])


class CompactSliderInterface:
    """A more compact slider interface for larger networks.
    
    Instead of showing all sliders at once, this interface allows
    selecting which layer/parameter to tune.
    """
    
    def __init__(self,
                 network: 'SimpleMLPNetwork',
                 X_data: np.ndarray,
                 y_data: np.ndarray,
                 max_sliders_visible: int = 20):
        """Initialize compact interface.
        
        Args:
            network: Neural network to tune
            X_data: Input data
            y_data: Target outputs
            max_sliders_visible: Maximum number of sliders to show at once
        """
        self.network = network
        self.X_data = X_data
        self.y_data = y_data
        self.max_sliders_visible = max_sliders_visible
        
        self.param_info = network.get_parameter_info()
        
        # Layer selector
        num_layers = len(network.weights)
        self.layer_selector = widgets.Dropdown(
            options=[(f'Layer {i}', i) for i in range(num_layers)],
            description='Layer:'
        )
        self.layer_selector.observe(self._on_layer_change, names='value')
        
        # Container for sliders
        self.slider_container = widgets.VBox([])
        
        # Output and metrics
        self.output_widget = widgets.Output()
        self.metrics_widget = widgets.HTML()
        
        # Build interface
        self._build_interface()
        self._on_layer_change(None)  # Initialize with first layer
    
    def _on_layer_change(self, change) -> None:
        """Update visible sliders when layer selection changes."""
        selected_layer = self.layer_selector.value
        
        # Get parameters for this layer
        layer_params = [p for p in self.param_info if p['layer'] == selected_layer]
        
        # Create sliders for this layer
        sliders = []
        for param in layer_params[:self.max_sliders_visible]:
            slider = widgets.FloatSlider(
                value=param['value'],
                min=-5.0,
                max=5.0,
                step=0.01,
                description=param['name'],
                continuous_update=False,
                readout_format='.3f',
                layout=widgets.Layout(width='400px')
            )
            sliders.append(slider)
        
        self.slider_container.children = sliders
    
    def _build_interface(self) -> None:
        """Build the interface layout."""
        left_panel = widgets.VBox([
            widgets.HTML("<h3>Parameter Selection</h3>"),
            self.layer_selector,
            self.slider_container
        ])
        
        right_panel = widgets.VBox([
            widgets.HTML("<h3>Visualization</h3>"),
            self.metrics_widget,
            self.output_widget
        ])
        
        self.interface = widgets.HBox([left_panel, right_panel])
    
    def display(self) -> None:
        """Display the interface."""
        display(self.interface)


def create_weight_heatmap(network: 'SimpleMLPNetwork', layer_idx: int = 0) -> Figure:
    """Create a heatmap visualization of weights for a specific layer.
    
    Args:
        network: Neural network
        layer_idx: Index of layer to visualize
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    weights = network.weights[layer_idx]
    im = ax.imshow(weights, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
    
    ax.set_xlabel(f'Layer {layer_idx + 1} neurons')
    ax.set_ylabel(f'Layer {layer_idx} neurons')
    ax.set_title(f'Weight Matrix - Layer {layer_idx}')
    
    plt.colorbar(im, ax=ax, label='Weight value')
    plt.tight_layout()
    
    return fig
