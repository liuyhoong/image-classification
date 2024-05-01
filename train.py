import numpy as np
import matplotlib.pyplot as plt
# 定义交叉熵损失函数
def cross_entropy_loss(outputs, targets):
    m = targets.shape[0]
    return -np.sum(targets * np.log(outputs + 1e-9)) / m

#定义计算准确率函数
def accuracy(outputs, labels):
    predictions = np.argmax(outputs, axis=1)
    labels = np.argmax(labels, axis=1)
    return np.mean(predictions == labels)

# 模型训练
def train_network(nn, X_train, y_train, X_val, y_val, epochs, batch_size, initial_lr, decay_rate=0.01):
    num_batches = len(X_train) // batch_size
    learning_rate = initial_lr
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # 混洗训练数据
        perm = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]
        epoch_losses = []

        # 迭代每个batch
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # 前向传播
            y_pred = nn.forward(X_batch)
            loss = nn.compute_loss(y_pred, y_batch)
            epoch_losses.append(loss)
            # 反向传播
            nn.backward(y_batch, learning_rate)

        # 计算平均训练损失
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        # 学习率衰减
        learning_rate = initial_lr / (1 + decay_rate * epoch)

        # 在验证集上评估模型
        y_val_pred = nn.forward(X_val)
        val_loss = nn.compute_loss(y_val_pred, y_val) / len(X_val) + nn.reg_lambda * sum(np.sum(w ** 2) for w in nn.weights) / (2*len(X_val))
        val_losses.append(val_loss)

        # 计算验证集准确率
        val_accuracy = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}, Validation Accuracy = {val_accuracy:.4f}")

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss during training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy on Validation Set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    return train_losses, val_losses, val_accuracies