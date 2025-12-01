# Neural Networks 🧠

## Overview
Neural Networks (also called Artificial Neural Networks or ANNs) are computing systems inspired by biological neural networks. They are the foundation of **Deep Learning** and can learn complex patterns from data.

## Architecture

### Basic Structure
```
Input Layer → Hidden Layer(s) → Output Layer
```

### Components

#### 1. Neurons (Nodes)
- Basic computational unit
- Receives inputs, applies weights, adds bias, applies activation

#### 2. Layers
- **Input Layer**: Receives raw data
- **Hidden Layers**: Process and transform data
- **Output Layer**: Produces final prediction

#### 3. Connections
- Each connection has a **weight**
- Weights are learned during training

## Mathematical Foundation

### Forward Propagation
```
z = Σ(wᵢ × xᵢ) + b
a = activation(z)
```
Where:
- w = weights
- x = inputs
- b = bias
- a = activation output

### Activation Functions

#### 1. Sigmoid
```
σ(x) = 1 / (1 + e⁻ˣ)
```
- Output: (0, 1)
- Use: Binary classification (output layer)

#### 2. ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
```
- Output: [0, ∞)
- Use: Hidden layers (most popular)

#### 3. Tanh (Hyperbolic Tangent)
```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```
- Output: (-1, 1)
- Use: Hidden layers

#### 4. Softmax
```
softmax(xᵢ) = eˣⁱ / Σeˣʲ
```
- Output: Probability distribution
- Use: Multi-class classification (output layer)

#### 5. Leaky ReLU
```
LeakyReLU(x) = max(0.01x, x)
```
- Solves "dying ReLU" problem

### Loss Functions

#### For Regression
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**

#### For Binary Classification
- **Binary Cross-Entropy**

#### For Multi-class Classification
- **Categorical Cross-Entropy**

### Backpropagation
Algorithm to update weights:
1. Calculate loss
2. Compute gradients using chain rule
3. Update weights: `w = w - learning_rate × gradient`

### Gradient Descent Variants

#### 1. Batch Gradient Descent
- Uses entire dataset
- Slow but stable

#### 2. Stochastic Gradient Descent (SGD)
- Uses one sample at a time
- Fast but noisy

#### 3. Mini-Batch Gradient Descent
- Uses small batches
- **Best of both worlds** (most common)

#### 4. Advanced Optimizers
- **Adam** (Adaptive Moment Estimation) - Most popular
- **RMSprop** - Good for RNNs
- **AdaGrad** - Adaptive learning rates

## Key Concepts

### Hyperparameters

#### Architecture
- **Number of layers**
- **Number of neurons per layer**
- **Activation functions**

#### Training
- **Learning rate** - Step size for weight updates
- **Batch size** - Samples per gradient update
- **Epochs** - Full passes through dataset
- **Optimizer** - Algorithm for weight updates

### Regularization Techniques

#### 1. Dropout
- Randomly "drop" neurons during training
- Prevents overfitting

#### 2. L1/L2 Regularization
- Add penalty for large weights
- L1: Lasso, L2: Ridge

#### 3. Early Stopping
- Stop training when validation loss stops improving

#### 4. Batch Normalization
- Normalize inputs to each layer
- Faster training, better performance

## Advantages
✅ Can learn complex non-linear patterns  
✅ Automatic feature extraction  
✅ Works with various data types (images, text, audio)  
✅ Highly flexible architecture  
✅ State-of-the-art performance on many tasks  

## Disadvantages
❌ Requires large amounts of data  
❌ Computationally expensive  
❌ "Black box" - hard to interpret  
❌ Many hyperparameters to tune  
❌ Prone to overfitting  
❌ Requires GPU for deep networks  

## Types of Neural Networks

### 1. Feedforward Neural Networks (FNN)
- Basic architecture
- Information flows forward only

### 2. Convolutional Neural Networks (CNN)
- For image processing
- Uses convolutional layers

### 3. Recurrent Neural Networks (RNN)
- For sequential data (text, time series)
- Has memory of previous inputs

### 4. Long Short-Term Memory (LSTM)
- Advanced RNN
- Better at long-term dependencies

### 5. Generative Adversarial Networks (GAN)
- Two networks competing
- Generate realistic data

### 6. Autoencoders
- Unsupervised learning
- Dimensionality reduction, denoising

## Use Cases
- 🖼️ **Image Classification** - Object recognition
- 🗣️ **Speech Recognition** - Voice assistants
- 📝 **Natural Language Processing** - Translation, chatbots
- 🎮 **Game Playing** - AlphaGo, chess engines
- 🚗 **Autonomous Vehicles** - Self-driving cars
- 🏥 **Medical Diagnosis** - Disease detection
- 💰 **Financial Forecasting** - Stock prediction
- 🎨 **Art Generation** - Style transfer, image generation

## Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix

### Regression
- R² Score
- MSE, RMSE, MAE

## Real-World Datasets
1. **MNIST** - Handwritten digits (28×28 images)
2. **CIFAR-10** - 60k images, 10 classes
3. **Fashion-MNIST** - Clothing images
4. **IMDB Reviews** - Sentiment analysis
5. **Time Series** - Stock prices, weather
6. **Kaggle Competitions** - Various challenges

## Common Problems & Solutions

### 1. Vanishing Gradients
- **Problem**: Gradients become too small
- **Solution**: Use ReLU, batch normalization, skip connections

### 2. Exploding Gradients
- **Problem**: Gradients become too large
- **Solution**: Gradient clipping, proper initialization

### 3. Overfitting
- **Problem**: Model memorizes training data
- **Solution**: Dropout, regularization, more data, early stopping

### 4. Slow Training
- **Problem**: Takes too long to train
- **Solution**: Use GPU, batch normalization, better optimizer (Adam)

## Frameworks & Libraries

### Python Libraries
- **TensorFlow** - Google's framework
- **Keras** - High-level API (now part of TensorFlow)
- **PyTorch** - Facebook's framework (very popular)
- **Scikit-learn** - MLPClassifier/MLPRegressor (basic NNs)

## Best Practices

### 1. Data Preprocessing
- Normalize/standardize features
- Handle missing values
- Augment data (for images)

### 2. Architecture Design
- Start simple, add complexity gradually
- Use ReLU for hidden layers
- Use appropriate output activation

### 3. Training
- Use Adam optimizer
- Monitor validation loss
- Use early stopping
- Save best model

### 4. Hyperparameter Tuning
- Grid search or random search
- Use cross-validation
- Start with learning rate tuning

## Files in This Folder
- `neural_networks_tutorial.ipynb` - Interactive Jupyter notebook
- `neural_network.py` - Implementation from scratch
- `keras_examples.py` - Using Keras/TensorFlow
- `pytorch_examples.py` - Using PyTorch
- `datasets/` - Sample datasets

## Resources
- [Scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Deep Learning Specialization (Andrew Ng)](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai Course](https://www.fast.ai/)
- [Neural Networks and Deep Learning (Book)](http://neuralnetworksanddeeplearning.com/)

## Next Steps
1. Open `neural_networks_tutorial.ipynb` in Jupyter
2. Understand forward and backward propagation
3. Build a simple network from scratch
4. Use Keras/PyTorch for practical applications
5. Experiment with different architectures
6. Try image classification with MNIST
7. Learn about CNNs and RNNs

---
**Happy Learning! 🚀**
