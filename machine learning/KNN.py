import math
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier


'''
p = 1 曼哈顿距离
p = 2 欧氏距离
p = inf 闵式距离minkowski_distance
'''


def L(x, y, p=2):
    # x,y都是数组p用来选择方法
    # x1 = [1, 1], x2 = [5,1]
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            # 欧氏距离 (xi-yi)的平方的和的二分之一次方
            sum = sum + math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1/p)
    else:
        return 0
x1 = [1, 1]
x2 = [5, 1]
x3 = [4, 4]
for i in range(1, 5):
    r = {'1-{}'.format(c): L(x1, c, p=i) for c in [x2, x3]}
    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    # 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
    print(min(zip(r.values(), r.keys())))


'''
python实现，遍历所有数据点，找出n个距离最近的点的分类情况，少数服从多数
'''
# 从iris加载数据
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['lable'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
print(df)
# df.iloc[0到100行，[第一列，第二列，最后一列]]
data = np.array(df.iloc[:100, [0, 1, -1]])
print(data)
# ：,：-1是去掉data最后一行赋值给X             :,-1 把data的最后一列数据赋值给y
X, y = data[:, :-1], data[:, -1]
# 划分测试集，训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''
kNN算法的核心思想是如果一个样本在特征空间中的k个最相邻的样本中的大多数属于某一个类别，则该样本也属于这个类别，并具有这个类别上样本的特性。该方法在确定分类
决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。 在KNN中，通过计算对象间距离来作为各个对象之间的非相似性指标，避免了对象之间的匹配问题
在这里距离一般使用欧氏距离或曼哈顿距离：
1）计算测试数据与各个训练数据之间的距离；
2）按照距离的递增关系进行排序；
3）选取距离最小的K个点；
4）确定前K个点所在类别的出现频率；
5）返回前K个点中出现频率最高的类别作为测试数据的预测分类。
'''


class KNN(object):
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        # parameter: n_neighbors 临近点个数
        # parameter: p 距离度量
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        knn_list = []
        for i in range(self.n):
            # 范数
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            # 注意(())
            knn_list.append((dist, self.y_train[i]))

        for i in range(self.n, len(self.X_train)):
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if knn_list[max_index][0] > dist:
                knn_list[max_index] = (dist, self.y_train[i])
        # 统计
        knn = [k[-1] for k in knn_list]
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs, key=lambda x: x)[-1]
        return max_count

    def score(self, X_test, y_test):
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)

clf = KNN(X_train, y_train)
clf.score(X_test, y_test)
test_point = [6.0, 3.0]
print('Test Point: {}'.format(clf.predict(test_point)))
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()

'''
直接调用SKLEARN封装好的
sklearn.neighbors.KNeighborsClassifier
n_neighbors: 临近点个数
p: 距离度量
algorithm: 近邻算法，可选{'auto', 'ball_tree', 'kd_tree', 'brute'}
weights: 确定近邻的权重
'''
print('sk')
clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')
score = clf_sk.score(X_test, y_test)
print(score)
