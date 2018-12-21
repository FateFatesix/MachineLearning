from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

'''
LR是经典的分类方法
回归模型：f(x) = （1+e的-wx次方）分之1
其中wx线性函数：wx=w0*x0+w1*x1+w2*x2+...+wn*xn,(x0=1)
'''


# creat data 还是这个套路
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # print(data)
    return data[:, :2], data[:, -1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


class LogisticReressionClassifier(object):
    """docstring for LogisticReressionClassifier"""
    def __init__(self, max_iter=200, learning_rate=0.01):
        # 迭代次数 与 学习率
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        # 经典
        return 1/(1+exp(-x))

    def data_matrix(self, X):
        data_mat = []
        for d in X:
            data_mat.append([1.0, *d])
        return data_mat

    def fit(self, X, y):
            # label = np.mat(y)
            data_mat = self.data_matrix(X)  # m * n
            # 生成 len(data_mat[0] 行 1列的0矩阵
            self.weights = np.zeros((len(data_mat[0]), 1), dtype=np.float32)

            for iter_ in range(self.max_iter):
                for i in range(len(X)):
                    # 经典
                    result = self.sigmoid(np.dot(data_mat[i], self.weights))
                    error = y[i] - result
                    # 通过结果 error 与 learningrate 调整权重
                    self.weights = self.weights + self.learning_rate * error * np.transpose([data_mat[i]])
            print('LogisticRegression Model(learning_rate={},max_iter={})'.format(self.learning_rate, self.max_iter))

    def score(self, X_test, y_test):
            right = 0
            X_test = self.data_matrix(X_test)
            for x, y in zip(X_test, y_test):
                result = np.dot(x, self.weights)
                if (result > 0 and y == 1) or (result < 0 and y == 0):
                    right += 1
            return right / len(X_test)

lr_clf = LogisticReressionClassifier()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)
# plt
x_ponits = np.arange(4, 8)
y_ = -(lr_clf.weights[1]*x_ponits + lr_clf.weights[0])/lr_clf.weights[2]
plt.plot(x_ponits, y_)
plt.scatter(X[:50, 0], X[:50, 1], label='0')
plt.scatter(X[50:, 0], X[50:, 1], label='1')
plt.legend()
plt.show()


'''
sklearn
sklearn.linear_model.LogisticRegression
solver参数决定了我们对逻辑回归损失函数的优化方法，有四种算法可以选择，分别是：
a) liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
b) lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
c) newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
d) sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。
'''

from sklearn.linear_model import LogisticRegression


clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
print(clf.coef_, clf.intercept_)
x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)
plt.scatter(X[:50, 0], X[:50, 1], label='0')
plt.scatter(X[50:, 0], X[50:, 1], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
