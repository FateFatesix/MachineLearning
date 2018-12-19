import numpy as np


class NN(object):
    """docstring for NN"""
    def __init__(self, arg):
        pass

    def train(self, X, y):
        # X 是一个N x D 的矩阵。y是大小为N的一维向量  Xtr 是训练集，X[i,::]作为验证集 （交叉验证）
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        # X 是一个N x D 的矩阵。我们想要去预测y的标签
        # 读取X的第一维度长度
        num_test = X.shape[0]
        # 保证矩阵type一致
        Ypred = np.zero(num_test, dtype=self.ytr.dtype)
        for i in range(num_test):
            # 寻找最邻近训练集与第i个测试集 使用曼哈顿距离 交叉验证
            distances = np.sum(np.abs(self.Xtr-X[i, ::], axis=1))
            # 返回f(x)值最小(即曼哈顿距离最小)的x
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]
        return Ypred
for i in range(2):
    p = np.array([[1,3,3],[23,3,4],[1,2,5]])
    print(p)
    print(p-p[i, ::])
print(np.sum(np.abs(p-p[i, ::])))