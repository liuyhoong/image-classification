from load_mnist import load_mnist
from model import one_hot, NeuralNetwork
from train import train_network
import numpy as np

def main_train_and_save():
    # 加载数据
    X_train, y_train = load_mnist('/Users/liuyanghong/Desktop/image_classification/data', kind='train')

    X_val, y_val = X_train[-10000:], y_train[-10000:]
    X_train, y_train = X_train[:-10000], y_train[:-10000]

    y_train = one_hot(y_train, num_classes=10)
    y_val = one_hot(y_val, num_classes=10)

    # 初始化神经网络
    nn = NeuralNetwork(layer_sizes=[784, 256, 64, 10], activation_funcs=['sigmoid', 'relu', 'softmax'])
    nn.reg_lambda = 0.001

    # 训练网络
    train_losses, val_losses, val_accuracies = train_network(
        nn, X_train, y_train, X_val, y_val, epochs=20, batch_size=100, initial_lr=0.001, decay_rate=0.01
    )

    # 保存模型参数
    np.save('model_weights.npy', nn.weights)  # 确保 NeuralNetwork 类有一个可以访问的 weights 属性


# 运行训练和保存模型
main_train_and_save()


def evaluate_on_test():
    # 加载测试数据
    X_test, y_test = load_mnist('/Users/liuyanghong/Desktop/image_classification/data', kind='t10k')
    y_test = one_hot(y_test, num_classes=10)

    # 加载模型参数
    model_weights = np.load('model_weights.npy', allow_pickle=True)

    # 重新初始化模型并加载权重
    nn = NeuralNetwork(layer_sizes=[784, 256, 64, 10], activation_funcs=['sigmoid', 'relu', 'softmax'])
    nn.weights = model_weights

    # 在测试集上评估模型
    y_test_pred = nn.forward(X_test)
    test_accuracy = np.mean(np.argmax(y_test_pred, axis=1) == np.argmax(y_test, axis=1))
    print(f"Test Accuracy: {test_accuracy:.4f}")


# 运行测试评估
evaluate_on_test()