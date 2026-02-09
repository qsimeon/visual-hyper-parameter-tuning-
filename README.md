# Interactive Neural Network Weight Tuning with Visual Sliders

> Explore neural network behavior by manipulating individual weights in real-time using interactive sliders and visualizations

This Jupyter notebook provides an interactive learning environment for understanding how individual weights in a neural network affect its output. Using IPython widgets, you can adjust each weight with sliders and immediately see how the network's predictions change. This hands-on approach makes abstract concepts like weight optimization, feature learning, and sparse auto-encoding tangible and intuitive. Perfect for students, educators, and anyone curious about the inner workings of neural networks!

## ✨ Features

- **Real-Time Weight Manipulation** — Interactive sliders for every individual weight in the neural network, allowing you to see instant feedback on how each parameter affects the model's output and behavior.
- **Visual Output Updates** — Dynamic matplotlib visualizations that update in real-time as you adjust weights, making it easy to understand the relationship between parameters and predictions.
- **Sparse Auto-Encoding Exploration** — Demonstrates concepts from interpretability research, including monosemantic features and sparse representations inspired by Anthropic's transformer circuits work.
- **Educational Interactive Interface** — Built with ipywidgets to create an intuitive, hands-on learning experience that demystifies neural network training and weight optimization.

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Lab or Jupyter Notebook
- Basic understanding of neural networks (helpful but not required)

### Setup

1. pip install numpy matplotlib ipywidgets
   - Installs core dependencies for numerical computation, visualization, and interactive widgets
2. jupyter labextension install @jupyter-widgets/jupyterlab-manager
   - Enables ipywidgets support in Jupyter Lab (may not be needed for newer versions)
3. Clone or download this repository to your local machine
   - Get the notebook file onto your computer
4. Navigate to the project directory in your terminal
   - Change to the folder containing the notebook
5. jupyter lab notebook.ipynb
   - Launches Jupyter Lab and opens the notebook

## 🚀 Usage

### Running Locally with Jupyter Lab

Launch the notebook on your local machine using Jupyter Lab for the full interactive experience with widget support.

```
# In your terminal:
cd path/to/notebook
jupyter lab notebook.ipynb

# Then in the notebook:
# 1. Run all cells sequentially (Shift + Enter)
# 2. Interact with the sliders that appear
# 3. Observe how the output visualization changes
```

**Output:**

```
Interactive sliders appear below the code cells, and matplotlib plots update in real-time as you adjust the weight values.
```

### Running in Google Colab

Upload and run the notebook in Google Colab for a zero-setup cloud-based experience. Perfect if you don't want to install anything locally.

```
# 1. Go to https://colab.research.google.com/
# 2. Click 'File' → 'Upload notebook'
# 3. Upload notebook.ipynb
# 4. Run all cells (Runtime → Run all)
# 5. Interact with the sliders

# Note: Dependencies are usually pre-installed in Colab,
# but you can run this cell if needed:
!pip install numpy matplotlib ipywidgets
```

**Output:**

```
The notebook runs in your browser with interactive widgets. All visualizations update dynamically as you manipulate the sliders.
```

### Experimenting with Weight Values

Try different weight configurations to understand their impact on network behavior and explore concepts like feature detection and sparse representations.

```
# After running all cells:
# 1. Start with all weights at their default values
# 2. Adjust one weight at a time and observe changes
# 3. Try setting most weights to zero (sparse representation)
# 4. Identify which weights have the most impact
# 5. Experiment with extreme values (very high/low)

# Example exploration:
# - Set weight_1 to maximum, others to minimum
# - Gradually increase weight_2 while keeping others constant
# - Create sparse patterns by zeroing out most weights
```

**Output:**

```
You'll gain intuition about which weights control which features, how sparsity affects representations, and how networks learn to encode information.
```

## 🏗️ Architecture

The notebook is structured as a progressive learning experience with 21 cells. It starts by importing dependencies and setting up the neural network architecture, then creates interactive widgets for each weight parameter, and finally connects these widgets to visualization functions that update in real-time. The architecture emphasizes hands-on experimentation over passive reading.

### File Structure

```
Notebook Flow:

┌─────────────────────────────────────┐
│  Cell 1-3: Imports & Setup          │
│  - numpy, matplotlib, ipywidgets    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Cell 4-8: Network Definition       │
│  - Define architecture              │
│  - Initialize weights               │
│  - Create forward pass function     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Cell 9-14: Widget Creation         │
│  - Slider for each weight           │
│  - Range and step configuration     │
│  - Layout and styling               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Cell 15-18: Visualization Logic    │
│  - Plot output function             │
│  - Connect widgets to updates       │
│  - Real-time rendering              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Cell 19-21: Interactive Demo       │
│  - Display all widgets              │
│  - Show initial visualization       │
│  - Enable user interaction          │
└─────────────────────────────────────┘
```

### Files

- **notebook.ipynb** — Main Jupyter notebook containing all code, widgets, and visualizations for interactive weight tuning.
- **README.md** — Documentation explaining the project, setup instructions, and usage examples.

### Design Decisions

- Used ipywidgets instead of other UI frameworks for seamless Jupyter integration and zero additional setup complexity.
- Structured notebook with progressive complexity, starting with simple concepts and building to interactive demonstrations.
- Chose matplotlib for visualizations due to its widespread familiarity and excellent Jupyter support.
- Implemented real-time updates using widget observers to provide immediate feedback on weight changes.
- Kept the neural network architecture simple (small number of weights) to make individual weight effects clearly visible.
- Incorporated sparse auto-encoding concepts to connect the interactive tool to cutting-edge interpretability research.

## 🔧 Technical Details

### Dependencies

- **numpy** (1.20.0+) — Provides efficient numerical computation for matrix operations, weight storage, and forward pass calculations.
- **matplotlib** (3.3.0+) — Creates dynamic visualizations that update in real-time as weights are adjusted via sliders.
- **ipywidgets** (7.6.0+) — Generates interactive sliders and UI controls that allow users to manipulate individual neural network weights.

### Key Algorithms / Patterns

- Forward propagation: Computes network output by multiplying inputs with weights and applying activation functions.
- Widget callback pattern: Uses observer pattern to trigger visualization updates whenever slider values change.
- Sparse representation: Demonstrates how setting most weights to zero can still produce meaningful outputs (sparse auto-encoding).
- Real-time rendering: Efficiently updates matplotlib plots without recreating the entire figure on each weight change.

### Important Notes

- Widget interactivity requires a live Jupyter kernel - static notebook viewers won't show interactive elements.
- Performance may degrade with very large networks; this demo is optimized for educational clarity over scale.
- The notebook explores concepts from Anthropic's monosemantic features research on transformer interpretability.
- Slider ranges are pre-configured but can be modified in the code to explore extreme weight values.

## ❓ Troubleshooting

### Sliders don't appear or aren't interactive

**Cause:** ipywidgets extension may not be properly installed or enabled in Jupyter Lab.

**Solution:** Run 'jupyter labextension install @jupyter-widgets/jupyterlab-manager' and restart Jupyter Lab. For Jupyter Notebook, try 'jupyter nbextension enable --py widgetsnbextension'.

### Visualizations don't update when moving sliders

**Cause:** The widget callback functions may not be properly connected, or the notebook kernel has crashed.

**Solution:** Restart the kernel (Kernel → Restart Kernel) and run all cells again from the beginning. Ensure you're running cells sequentially.

### ImportError: No module named 'ipywidgets'

**Cause:** The ipywidgets package is not installed in your Python environment.

**Solution:** Install the package using 'pip install ipywidgets' in your terminal. If using conda, try 'conda install -c conda-forge ipywidgets'.

### Plots appear but don't update smoothly

**Cause:** Matplotlib backend may not be optimized for interactive use, or too many updates are queued.

**Solution:** Add '%matplotlib widget' at the top of the notebook for better interactivity. Alternatively, use '%matplotlib notebook' for older Jupyter versions.

### Notebook works locally but not in Google Colab

**Cause:** Colab may have different widget rendering or require specific matplotlib backends.

**Solution:** Ensure you run all cells in order. Colab usually supports ipywidgets natively, but you may need to use '%matplotlib inline' instead of other backends.

---

This README was generated to help learners understand neural network weight optimization through hands-on experimentation. The project bridges the gap between theoretical understanding and practical intuition by making abstract mathematical concepts visually tangible. The sparse auto-encoding concepts referenced are inspired by cutting-edge interpretability research from Anthropic's Transformer Circuits thread (https://transformer-circuits.pub/2023/monosemantic-features), which explores how neural networks learn interpretable features. This notebook serves as an accessible entry point to these advanced topics, making them approachable for students and enthusiasts at all levels.