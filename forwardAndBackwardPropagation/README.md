# Neural Network Implementation with Forward and Backward Propagation

## Abstract

This study presents a from-scratch implementation of a feedforward neural network using forward and backward propagation algorithms for both classification and regression tasks. The implementation includes multiple activation functions (ReLU, Sigmoid, Tanh, Linear, Softmax), gradient-based optimization with mini-batch support, and early stopping mechanisms. Experimental validation was conducted on synthetic datasets, achieving 87% accuracy on binary classification tasks and R² score of 0.9443 on regression problems, demonstrating the effectiveness of the implemented algorithms.

**Keywords:** Neural Networks, Forward Propagation, Backward Propagation, Gradient Descent, Machine Learning

## 1. Introduction

### 1.1 Background

Artificial neural networks have become fundamental tools in machine learning, capable of learning complex patterns from data through iterative optimization processes. The core mechanisms underlying neural network training are forward propagation and backward propagation algorithms, which enable the network to learn by adjusting weights and biases based on prediction errors.

### 1.2 Motivation

While numerous neural network frameworks exist, understanding the fundamental algorithms through implementation provides deeper insights into the mathematical foundations and computational processes. This study implements a neural network from scratch to demonstrate the core principles of forward and backward propagation.

### 1.3 Objectives

The primary objectives of this research are:
- Implement forward propagation for computing activations through network layers
- Develop backward propagation using the chain rule for gradient computation
- Support multiple activation functions and loss functions
- Validate the implementation on classification and regression tasks
- Demonstrate convergence and performance metrics

### 1.4 Scope

This implementation focuses on fully connected feedforward networks with support for:
- Binary and multi-class classification
- Regression tasks
- Multiple activation functions (ReLU, Sigmoid, Tanh, Linear, Softmax)
- Mini-batch gradient descent optimization
- Early stopping for regularization

## 2. Methods

### 2.1 Neural Network Architecture

The implemented neural network consists of:
- **Input Layer**: Receives feature vectors
- **Hidden Layers**: Apply linear transformations followed by activation functions
- **Output Layer**: Produces predictions using task-specific activation functions

### 2.2 Forward Propagation Algorithm

Forward propagation computes activations layer by layer:

For each layer l:
```
z^[l] = W^[l] · a^[l-1] + b^[l]
a^[l] = g(z^[l])
```

Where:
- W^[l]: Weight matrix for layer l
- b^[l]: Bias vector for layer l
- g(): Activation function
- a^[0] = X (input features)

### 2.3 Activation Functions

The implementation includes:

**ReLU**: f(x) = max(0, x)
- Used in hidden layers for non-linearity
- Gradient: f'(x) = 1 if x > 0, else 0

**Sigmoid**: f(x) = 1/(1 + e^(-x))
- Used for binary classification output
- Gradient: f'(x) = f(x)(1 - f(x))

**Tanh**: f(x) = tanh(x)
- Alternative activation for hidden layers
- Gradient: f'(x) = 1 - tanh²(x)

**Softmax**: f(x_i) = e^(x_i) / Σe^(x_j)
- Used for multi-class classification
- Provides probability distribution over classes

**Linear**: f(x) = x
- Used for regression output layers
- Gradient: f'(x) = 1

### 2.4 Backward Propagation Algorithm

Backward propagation computes gradients using the chain rule:

**Output Layer Error**:
- Classification: δ^[L] = a^[L] - y
- Regression: δ^[L] = a^[L] - y

**Hidden Layer Errors**:
```
δ^[l] = (W^[l+1])^T δ^[l+1] ⊙ g'(z^[l])
```

**Weight and Bias Gradients**:
```
∂L/∂W^[l] = (1/m) δ^[l] (a^[l-1])^T
∂L/∂b^[l] = (1/m) Σδ^[l]
```

### 2.5 Loss Functions

**Binary Cross-Entropy** (Classification):
```
L = -(1/m) Σ[y log(ŷ) + (1-y) log(1-ŷ)]
```

**Categorical Cross-Entropy** (Multi-class):
```
L = -(1/m) Σ Σ y_ij log(ŷ_ij)
```

**Mean Squared Error** (Regression):
```
L = (1/m) Σ(y - ŷ)²
```

### 2.6 Optimization

**Mini-batch Gradient Descent**:
- Data shuffled at each epoch
- Weights updated per batch: W = W - α∇W
- Gradient clipping applied for stability

**Weight Initialization**:
- Xavier initialization for regression: σ = √(2/(n_in + n_out))
- He initialization for ReLU: σ = √(2/n_in)

**Early Stopping**:
- Monitor loss on training set
- Stop training if no improvement for 50 epochs

### 2.7 Experimental Setup

**Classification Dataset**:
- 1000 samples, 20 features
- 15 informative, 5 redundant features
- Binary classification task
- Train/test split: 80/20

**Regression Dataset**:
- 1000 samples, 10 features
- 8 informative features
- Continuous target variable
- Train/test split: 80/20

**Network Architectures**:
- Classification: [20, 16, 8, 1] with ReLU → Sigmoid
- Regression: [10, 12, 6, 1] with Tanh → Linear

**Training Parameters**:
- Learning rate: 0.01
- Epochs: 1000
- Batch size: Full batch
- Optimization: Gradient descent with clipping

## 3. Results

### 3.1 Classification Performance

**Training Progress**:
- Initial loss: 0.8539
- Final loss: 0.3335
- Initial accuracy: 49.88%
- Final accuracy: 87.12%

**Test Performance**:
- Test accuracy: 87.00%
- Consistent with training performance, indicating no overfitting

**Convergence**:
- Steady loss decrease over 1000 epochs
- Accuracy improvement: 37+ percentage points
- Stable training without gradient explosion

### 3.2 Regression Performance

**Training Progress**:
- Initial loss: 1.2302
- Final loss: 0.0523
- Rapid convergence within first 200 epochs

**Test Performance**:
- Test MSE: 0.0463
- Test R² score: 0.9443 (94.43% variance explained)
- Strong predictive performance

### 3.3 Training Dynamics

**Loss Convergence**:
- Classification: Exponential decay pattern
- Regression: Fast initial drop, then gradual improvement
- No signs of overfitting in either task

**Gradient Stability**:
- Gradient clipping prevented explosion
- Stable parameter updates throughout training
- Consistent convergence across multiple runs

### 3.4 Computational Performance

**Memory Efficiency**:
- Mini-batch support for large datasets
- Efficient matrix operations using NumPy
- Reasonable computational complexity

**Training Time**:
- Fast convergence for both tasks
- Early stopping reduced unnecessary computation
- Suitable for medium-scale problems

## 4. Discussion

### 4.1 Algorithm Effectiveness

The implemented neural network successfully demonstrates the fundamental principles of forward and backward propagation. The achieved performance metrics (87% classification accuracy, 94.43% regression R²) validate the correctness of the implementation and effectiveness of the algorithms.

### 4.2 Forward Propagation Analysis

The forward propagation implementation efficiently computes activations layer by layer, properly handling different activation functions. The modular design allows easy extension to different network architectures and activation functions.

### 4.3 Backward Propagation Analysis

The backward propagation correctly implements the chain rule for gradient computation. The automatic differentiation through the network layers demonstrates proper understanding of the mathematical foundations. Gradient clipping prevents numerical instabilities commonly encountered in deep networks.

### 4.4 Optimization Performance

The gradient descent optimization with appropriate learning rates achieves stable convergence. The weight initialization strategies (Xavier/He) contribute to training stability. Early stopping effectively prevents overfitting in regression tasks.

### 4.5 Limitations and Considerations

**Scalability**: The current implementation is suitable for small to medium-scale problems. For larger networks, more sophisticated optimizers (Adam, RMSprop) would be beneficial.

**Architecture Constraints**: Limited to fully connected layers. Convolutional or recurrent architectures would require additional implementation.

**Regularization**: Currently implements only early stopping. Dropout, L1/L2 regularization could improve generalization.

### 4.6 Practical Applications

This implementation serves as:
- Educational tool for understanding neural network fundamentals
- Baseline for comparing with advanced frameworks
- Foundation for extending to more complex architectures
- Demonstration of mathematical concepts in practice

### 4.7 Future Improvements

**Optimization Enhancements**:
- Adaptive learning rate methods
- Momentum-based optimization
- Learning rate scheduling

**Regularization Techniques**:
- Dropout layers
- Batch normalization
- Weight decay

**Architecture Extensions**:
- Convolutional layers
- Recurrent connections
- Attention mechanisms

## 5. Conclusion

This study successfully implemented a neural network with forward and backward propagation algorithms from scratch, demonstrating solid understanding of the underlying mathematical principles. The implementation achieved competitive performance on both classification (87% accuracy) and regression (94.43% R²) tasks, validating the correctness of the algorithms.

**Key Contributions**:
1. Complete implementation of forward/backward propagation
2. Support for multiple activation functions and loss functions
3. Efficient mini-batch gradient descent optimization
4. Experimental validation on synthetic datasets
5. Comprehensive performance analysis and visualization

**Educational Value**:
The implementation provides clear insights into neural network training dynamics, gradient computation, and optimization processes. The modular design facilitates understanding of individual components and their interactions.

**Performance Summary**:
- Binary classification: 87% test accuracy
- Regression: 94.43% variance explained (R²)
- Stable convergence without overfitting
- Efficient computational performance

The results demonstrate that fundamental neural network algorithms, when properly implemented, can achieve strong performance on standard machine learning tasks. This work provides a solid foundation for understanding more advanced neural network architectures and training techniques.

## References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. Nielsen, M. (2015). *Neural Networks and Deep Learning*. Determination Press.
3. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
4. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
5. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*, 9, 249-256.

---

**Author**: Ural Altan Bozkurt  
**Course**: YZM212 Machine Learning  
**Date**: May 31, 2025  
**Institution**: Ankara Üniversitesi
