import os
import numpy as np


def load_mnist(path, kind='train'):
    """加载Fashion-MNIST数据的函数"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')
    # 读取文件（二进制读取方法）
    with open(labels_path, 'rb') as lbpath:
        # 读取标签数据，跳过前8个字节的头信息（前八个数据为描述文件元数据）
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as imgpath:
        # 读取图像数据，跳过前16个字节的头信息（同理），这里使用reshape函数将图像数据重新形状为(len(labels), 784)，其中len(labels)表示图像数量，784表示每个图像的像素值展平为一维数组。
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    # 返回加载的图像数据和标签数据作为元组
    return images, labels
