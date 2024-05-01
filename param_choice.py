import numpy as np
from model import NeuralNetwork
from model import one_hot
from load_mnist import load_mnist
from train import train_network

def search_hyperparameters(layer_size_options, learning_rate_options, reg_lambda_options, epochs, batch_size):
    best_val_accuracy = -np.inf
    best_hyperparameters = {}

    # 遍历所有超参数组合
    for layer_sizes in layer_size_options:
        for learning_rate in learning_rate_options:
            for reg_lambda in reg_lambda_options:
                print(
                    f"Testing with layers: {layer_sizes}, learning rate: {learning_rate}, regularization: {reg_lambda}")

                # 初始化神经网络
                nn = NeuralNetwork(
                    layer_sizes=layer_sizes,
                    activation_funcs=['sigmoid', 'relu', 'softmax'],
                    loss_func='cross_entropy'
                )
                nn.reg_lambda = reg_lambda

                # 加载数据
                X_train, y_train = load_mnist('/Users/liuyanghong/Desktop/image_classification/data', kind='train')

                # 保留一部分训练数据用于验证
                X_val, y_val = X_train[-10000:], y_train[-10000:]
                X_train, y_train = X_train[:-10000], y_train[:-10000]

                # 转换标签为one-hot编码
                y_train = one_hot(y_train, num_classes=10)
                y_val = one_hot(y_val, num_classes=10)

                # 训练网络
                train_losses, val_losses, val_accuracies = train_network(
                    nn, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate
                )

                # 评估当前超参数组合
                current_val_accuracy = val_accuracies[-1]
                if current_val_accuracy > best_val_accuracy:
                    best_val_accuracy = current_val_accuracy
                    best_hyperparameters = {
                        'layer_sizes': layer_sizes,
                        'learning_rate': learning_rate,
                        'reg_lambda': reg_lambda
                    }
                    print(f"New best validation accuracy: {best_val_accuracy:.4f}")

    # 输出最佳的超参数组合和对应的验证集准确率
    print("Best Hyperparameters found:")
    print(f"Layers: {best_hyperparameters['layer_sizes']}")
    print(f"Learning Rate: {best_hyperparameters['learning_rate']}")
    print(f"Regularization Lambda: {best_hyperparameters['reg_lambda']}")
    print(f"Validation Accuracy: {best_val_accuracy:.4f}")


# 进行选择
layer_size_options = [[784, 256, 64, 10], [784, 128, 64, 10]]
learning_rate_options = [0.01, 0.001, 0.005]
reg_lambda_options = [0.01, 0.001]
search_hyperparameters(layer_size_options, learning_rate_options, reg_lambda_options, epochs=5, batch_size=100)