# sklearn 数据处理

## 1. 数据集转换

### 1.1. 预处理数据

#### 1.1.1. 标准化或平均去除和方差缩放

​	数据标准化是 sklearn 中实现的许多机器学习估计器的常见要求；标准化操作就是减去每个特征的平均值然后除以方差（或标准差）；形成标准正态分布（具有零均值和单位方差的高斯分布）。

```python
# scale 函数提供一种快速简单的方法来在单个阵列数据集上执行此操作：
from sklearn import preprocessing
import numpy as np

X = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])

X_scaled = preprocessing.scale(X)
'''
结果为：
array([[ 0.  ..., -1.22...,  1.33...],
       [ 1.22...,  0.  ..., -0.26...],
       [-1.22...,  1.22..., -1.06...]])
'''

# preprocessing 模块进一步提供了一个实用程序类 StandarScaler，它实现 Transformer API 来计算训练集上的平均值和标准偏差，以便能够稍后在测试集上重新应用相同的变换。
# 计算训练集上的平均值和标准差
scaler = preprocessing.StandardScaler().fit(X_train)
# 将训练集的平均值和标准差应用于测试集
scaler.transform(X_test)
```

##### 1.1.1.1. 将特征数据缩放到指定范围

​	替代的

## 2. 模型选择与评估



