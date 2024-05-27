## Clustering" 是指将数据集中的对象分组成具有相似性的子集的过程。在这个过程中，被分到同一组的对象之间应该有较高的相似性，而不同组之间的对象应该有较低的相似性。这样的划分有助于揭示数据集内部的结构，并且能够帮助我们理解数据的特征、模式和分布。

Clustering 是无监督学习的一种方法，因为它不需要任何预先标记的数据。相反，它根据数据本身的特征来确定分组。
* 在 Clustering 中，通常有两个关键概念：
  - 相似性度量（The distance between two points and in n-dimensional space can be measured in different ways:）：用于衡量数据对象之间的相似性。相似性度量可以根据具体的应用选择，例如欧氏距离（Euclidean distance）、曼哈顿距离（Manhattan distance）、Minkowski distance or 余弦相似度(Cosine similarity )等。
  - 聚类算法：用于将数据对象划分成不同的组。常见的聚类算法包括 K-means、层次聚类、DBSCAN 等。
* Clustering 的目标是使得同一组内的对象尽可能相似，而不同组之间的对象尽可能不同，从而帮助我们理解数据集的结构和特征。
* The similarity of sets A and B can be measured by the Jaccard index.Jaccard 指数是一种用于比较两个集合之间相似性的指标，通常用于评估分类模型或聚类模型的性能。Jaccard 指数可以用以下公式表示：
  $$J(A, B) = \frac{{\left| A \cap B \right|}}{{\left| A \cup B \right|}}$$
  - Jaccard 指数的取值范围是 [0, 1]，其中 0 表示两个集合没有共同元素，1 表示两个集合完全相同。因此，Jaccard 指数越接近于 1，表示两个集合越相似。
  - 在机器学习中，Jaccard 指数通常用于评估分类模型的性能。在二分类任务中，可以将模型的预测结果与真实标签构成两个集合，然后计算它们之间的 Jaccard 指数来衡量模型的准确性。Jaccard 指数也可以用于评估聚类模型的性能，用于比较聚类结果与真实类别之间的相似性。

* 常见的应用场景包括：
- 客户分群：根据客户的行为、购买历史等信息将客户分成不同的组，以便于个性化营销和服务。
- 文档聚类：将文档按照主题或内容相似性进行分组，以便于文本分类、信息检索等任务。
- 图像分割：将图像中的像素根据颜色、纹理等特征分成不同的区域，用于目标检测、图像分析等应用。

### Clustering 的三种算法
#### 1,K-Means clustering
* 算法原理：K-means 算法通过将数据分成 K 个簇，并将每个数据点分配到最近的簇，然后通过计算每个簇的中心来更新簇的分配。这个过程不断迭代，直到簇的中心不再发生变化或达到最大迭代次数为止。
* 参数设置：
  - n_clusters：要分成的簇的数量 K。
  - init：初始化中心的方法，常见的有 "k-means++"（默认）、"random" 等。
  - max_iter：最大迭代次数，决定算法迭代的次数。
  - random_state：随机种子，用于初始化中心点。
* K值选择的几个方法(有几种方法可以评估 K-means 模型的聚类质量以确定最合适的簇数)：
   - 肘部法（Elbow Method）：这是最常用的方法之一。它涉及绘制不同 K 值下的误差平方和（SSE）的图表，并观察 SSE 随着 K 值增加而减少的速度。当增加 K 值不再显著减少 SSE 时，这个点就是所谓的“肘部”。在肘部之后，增加 K 值对 SSE 的改善不再那么明显，因此可以选择肘部对应的 K 值作为最佳的簇数。
   - 轮廓系数（Silhouette Score）：轮廓系数综合了簇内距离和簇间距离的信息，可以用来衡量聚类的紧密度和分离度。具体来说，轮廓系数在 [-1, 1] 的范围内，值越接近 1 表示聚类越好，值越接近 -1 表示聚类越差。因此，可以通过计算不同 K 值下的轮廓系数来选择最合适的簇数。
   - 间隔统计量（Gap Statistics）：间隔统计量是一种比较聚类结果与随机数据集之间差异的方法。它计算了实际数据的聚类结果与在随机数据集上的聚类结果之间的间隔，然后将其与一些随机生成的参考数据集进行比较。间隔统计量的最大值通常对应于最佳的簇数。
   - 轮廓图（Silhouette Plot）：轮廓图是一种可视化方法，用于直观地比较不同 K 值下的轮廓系数。它通过绘制每个数据点的轮廓系数来显示每个簇的紧密度和分离度，从而帮助选择最合适的簇数。
     [Repo](https://musical-disco-22w162z.pages.github.io/sessions/23_Clustering.html)

#### 2,Hierarchical clustering(层次聚类算法):
* 算法原理：层次聚类算法通过计算数据对象之间的相似性来构建一个层次结构，然后根据这个层次结构来划分数据。最常见的层次聚类是凝聚式（agglomerative）聚类，它从每个数据点作为单独的簇开始，并逐渐将相似的簇合并，直到达到指定的簇的数量或者达到某个相似度的阈值。(此算法不依赖初始化)
* 参数设置：
  - n_clusters：要分成的簇的数量（仅在凝聚式聚类中使用）。
  - linkage：链接标准，用于衡量两个簇之间的距离。常见的包括 "ward"、"complete"、"average" 等。
  - affinity：用于计算簇之间的距离。常见的包括 "euclidean"（默认）、"manhattan"、"cosine" 等。
* K值选择：常有以下方法（示例在clustering 脚本中）
  - 树状图（Dendrogram）：树状图是层次聚类的一个重要工具，它通过可视化地显示数据点的合并过程来帮助确定最佳的簇数。在树状图中，每个分支的垂直轴表示两个簇之间的距离或相似度。你可以通过寻找“最大的垂直距离”（即高度差异最大）来决定最佳的簇数。在这条垂直线上切割树状图可以得到不同的簇数。例如，如果树状图在某个高度突然有一个较大的间隙，这表明在此高度切割树状图可能是最佳的簇数选择。
  - 轮廓系数（Silhouette Score）：轮廓系数综合了簇内距离和簇间距离的信息，可以用来衡量聚类的紧密度和分离度。对于不同的簇数 k，计算其对应的轮廓系数。最佳的簇数通常对应于最高的平均轮廓系数。可以使用 sklearn.metrics.silhouette_score 函数计算轮廓系数。
  - 距离统计量（Inconsistency Method）：通过计算每次合并的簇之间的距离变化来确定最佳的簇数。可以使用 scipy.cluster.hierarchy.inconsistent 函数来计算不一致性统计量，并在距离变化较大处进行切割以确定簇数。

 #### 3，DBSCAN 聚类算法：
 * 算法原理：DBSCAN（Density-Based Spatial Clustering of Applications with Noise）通过定义核心点、边界点和噪声点来构建聚类。它从一个核心点开始，找到密度可达的所有点，并将它们分配到同一个簇中，然后通过密度可达性来扩展簇，直到所有点都被访问。
 * 参数设置：
  - eps：邻域半径，用于确定核心点的邻域大小。
  - min_samples：核心点所需的最小样本数。
  - metric：用于计算点之间距离的度量方法。
  - algorithm：用于计算邻域的算法，常见的有 "auto"、"ball_tree"、"kd_tree"、"brute"。
* K值选择：
  - 计算每个点到其最近 k 个邻居的距离，k 通常设置为 min_samples 值。对所有点按降序排列这些距离。绘制 k-距离图，横轴是点的索引，纵轴是对应的 k-距离。寻找图中的“膝点”或“拐点”，该点通常对应于较好的 eps 值。
  - 根据行业经验