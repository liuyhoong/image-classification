# image-classification
首先，打开param-choice文件中的search_hyperparameters函数下的初始化神经网络那里修改activation_funcs列表，将列表中前两个函数修改为两个激活函数，两个都可以选择‘relu’或‘sigmoid’。
然后在main函数中设置，调节参数的列表，layer_size_options为各个层的维度，例：[784, 256, 64, 10]表示输入层的维度为784，第一个隐藏层为256，第二个隐藏层为64，输出为10。然后运行main函数，最后会输出一个最好的超参数组合。
然后使用这一组合超参数进去train_and_test中进行训练和测试，只需将最好的参数带入相关位置即可！

注意：要将load_mnist函数中的path换成本地电脑的路径
