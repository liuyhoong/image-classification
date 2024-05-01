import numpy as np
import matplotlib.pyplot as plt
class NeuralNetwork:
    def __init__(self, layer_sizes, activation_funcs, loss_func='cross_entropy'):
        self.layer_sizes = layer_sizes
        self.activation_funcs = activation_funcs
        self.loss_func = loss_func
        self.weights = []
        self.biases = []
        self.activations = []

        # 初始化权重和偏置
        for i in range(len(layer_sizes)-1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            bias = np.zeros(layer_sizes[i+1])
            self.weights.append(weight)
            self.biases.append(bias)

    def activate(self, z, func):
        if func == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif func == 'relu':
            return np.maximum(0, z)
        elif func == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return z

    def derivative(self, z, func):
        if func == 'sigmoid':
            sig = self.activate(z, 'sigmoid')
            return sig * (1 - sig)
        elif func == 'relu':
            return (z > 0).astype(float)
        return 1

    def forward(self, x):
        self.activations = [x]
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = self.activate(z, self.activation_funcs[i])
            self.activations.append(a)
        # 最后一层使用softmax函数
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        a = self.activate(z, 'softmax')
        self.activations.append(a)
        return self.activations[-1]

    def compute_loss(self, y_pred, y_true):
        if self.loss_func == 'cross_entropy':
            return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
        return 0

    def backward(self, y_true, learning_rate):
        m = y_true.shape[0]
        error = self.activations[-1] - y_true

        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.activations[i].T, error) / m
            dB = error.sum(axis=0) / m
            if i > 0:
                error = np.dot(error, self.weights[i].T) * self.derivative(self.activations[i], self.activation_funcs[i-1])

            # L2 正则化的影响
            dW += self.reg_lambda * self.weights[i]  # 加入正则化项

            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * dB

# 转换标签为one-hot编码的辅助函数
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]