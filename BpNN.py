import random
import math
import matplotlib.pyplot as plt

'''
三层神经网络模型，包含：输入层、隐层、输出层
'''
class BP:
    '''
    构造函数，初始化三层神网络的各参数

    Args:
        x_count: 输入层神经元个数
        mid_count: 隐层神经元个数
        y_count: 输出层神经元个数
        eta: 学习率
        train_count: 最大训练次数
        precision: 误差精度
    '''
    def __init__(self, x_count, mid_count, y_count, eta=0.3, train_count=100, precision=0.00001):
        self.x_count = x_count
        self.mid_count = mid_count
        self.y_count = y_count
        self.eta = eta
        self.train_count = train_count
        self.precision = precision

        # 输入层到隐层的权值
        self.V = []
        for i in range(0, x_count):
            temp = []
            for j in range(0, mid_count):
                temp.append(2*random.random() - 1)
            self.V.append(temp)

        # 输入层到隐层的阈值
        self.gamma = []
        for i in range(0, mid_count):
            self.gamma.append(2*random.random() - 1)

        # 隐层到输出层的权值
        self.W = []
        for i in range(0, mid_count):
            temp = []
            for j in range(0, y_count):
                temp.append(2*random.random() - 1)
            self.W.append(temp)

        # 隐层到输出层的阈值
        self.beta = []
        for i in range(0, y_count):
            self.beta.append(2*random.random() - 1)

    '''
    神经网络训练函数

    Args:
        X: 列表，输入数据
        Y: 列表，实际输出数据
    '''
    def train(self, X, Y):
        if len(X) != len(Y):
            print("Error: len(X) and len(Y) is unequal!!!")
            return

        for i in range(self.train_count):
            E = [] # 每一组数据的误差
            # 遍历每一组输入数据
            for j in range(len(X)):
                # 计算预测值
                y_predict, mid_output = self.compute_y(X[j])

                # 计算当前样例(组)的均方误差
                e = 0.0
                mid2y_g = [] # 隐层到输出层的梯度项
                for k in range(self.y_count):
                    # 计算输出层第k个神经元的误差
                    e += pow(y_predict[k] - Y[k], 2)
                E.append(e/2)

                # 计算隐层到输出层的梯度项
                mid2y_g = []
                for k in range(self.y_count):
                    # 计算输出层第k个神经元对应的，隐层到输出层的梯度项
                    mid2y_g.append(y_predict[k] * (1 - y_predict[k]) * (Y[k] - y_predict[k]))

                # 计算输入层到隐层的梯度项
                x2mid_g = []
                for k in range(self.mid_count):
                    temp = 0
                    for l in range(self.y_count):
                        temp += self.W[k][l] * mid2y_g[l]
                    # 计算隐层第k个神经元对应的，输入层到隐层的梯度项
                    x2mid_g.append(mid_output[k] * (1 - mid_output[k]) * temp)

                # 更新隐层到输出层的权值和阈值
                for k in range(self.mid_count):
                    for l in range(self.y_count):
                        self.W[k][l] += self.eta * mid2y_g[l] * mid_output[k]
                for k in range(self.y_count):
                    self.beta[k] -= self.eta * mid2y_g[k]

                # 更新输入层到隐层的权值和阈值
                for k in range(self.x_count):
                    for l in range(self.mid_count):
                        self.V[k][l] += self.eta * x2mid_g[l] * X[j][k]
                for k in range(self.mid_count):
                    self.gamma[k] -= self.eta * x2mid_g[k]

            # 计算累积误差
            E_sum = 0.0
            for e in E:
                E_sum += e
            E_sum /= len(E)
            print(E_sum)

            # 如果累计误差小于设定的误差精度，则停止训练
            if E_sum < self.precision:
                break
    '''
    Sigmoid激活函数

    Args:
        x

    Returns:
        y: sigmoid(x)
    '''
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    '''
    计算一组预测值

    Args:
        x: 列表，一组多元或一元的输入数据

    Returns:
        y: 列表，一组多元或一元的输出数据
        mid_output: 列表，隐层的输出数据
    '''
    def compute_y(self, x):
        # 计算隐层输入
        mid_input = []
        for i in range(self.mid_count):
            temp = 0
            for j in range(self.x_count):
                temp += self.V[j][i] * x[j]
            mid_input.append(temp)

        # 计算隐层输出
        mid_output = []
        for i in range(self.mid_count):
            mid_output.append(self.sigmoid(mid_input[i] - self.gamma[i]))

        # 计算输出层的输入
        y_input = []
        for i in range(self.y_count):
            temp = 0
            for j in range(self.mid_count):
                temp += self.W[j][i] * mid_output[j]
            y_input.append(temp)

        # 计算输出层的输出
        y = []
        for i in range(self.y_count):
            y.append(self.sigmoid(y_input[i] - self.beta[i]))

        return (y, mid_output)

    '''
    神经网络预测函数

    Args:
        X: 列表，输入数据

    Returns:
        Y_predict: 列表，预测输出数据
    '''
    def predict(self, X):
        Y_predict = []
        for x in X:
            y_predict, _ = self.compute_y(x)
            Y_predict.append(y_predict)
        return Y_predict



