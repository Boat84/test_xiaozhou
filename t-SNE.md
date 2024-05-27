## t-distributed stochastic neighbor embedding（t-SNE）（t 分布随机邻域嵌入）定义：
* 是一种用于降维的非线性技术，常用于高维数据的可视化。t-SNE能够将高维数据投影到二维或三维空间中，同时尽可能保留原始数据的局部结构，从而使得数据中的模式和聚类变得更容易观察。t-SNE由Laurens van der Maaten和Geoffrey Hinton在2008年提出。

## 工作原理（t-SNE的核心思想）：

   ### 1. 计算高维空间中的相似度
* 对于每对高维数据点 \( x_i \) 和 \( x_j \)，计算高维空间中的相似度。相似度用条件概率\( p_{j|i} \) 表示，表示在给定点 \( x_i \) 的条件下，点 \( x_j \) 被选择作为邻居的概率。

$$\[ 
p_{j|i} = \frac{\exp\left(-\|x_i - x_j\|^2 / 2\sigma_i^2\right)}{\sum_{k \neq i} \exp\left(-\|x_i - x_k\|^2 / 2\sigma_i^2\right)} 
\]$$

为对称化，定义联合概率：

$$\[ 
p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N} 
\]$$

### 2. 计算低维空间中的相似度

* 在低维空间中，使用t分布（t-distribution）计算每对点 \( y_i \) 和 \( y_j \) 的相似度。相似度用条件概率 \( q_{ij} \) 表示：

$$\[ 
q_{ij} = \frac{\left(1 + \|y_i - y_j\|^2\right)^{-1}}{\sum_{k \neq l} \left(1 + \|y_k - y_l\|^2\right)^{-1}} 
\]$$

### 3. 最小化差异

* 通过最小化高维和低维空间中概率分布之间的Kullback-Leibler散度（KL散度）来优化低维空间中的点 \( y_i \) 的位置：

$$ \[ 
KL(P||Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}} 
\]$$


## 应用:t-SNE广泛应用于高维数据的可视化，包括但不限于：
* 图像数据的降维可视化
* 文本数据的降维可视化
* 基因表达数据的降维可视化
* 手写数字数据（如MNIST）的降维可视化

### 示例代码
* 下面是使用t-SNE对手写数字数据集（如Scikit-learn中的digits数据集）进行降维并可视化的示例代码：
```import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# 加载digits数据集
digits = datasets.load_digits()
X = digits.data
y = digits.target

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, init='random', random_state=0)
X_tsne = tsne.fit_transform(X)

# 可视化降维后的数据
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title('t-SNE visualization of the digits dataset')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()
```

