import numpy as np

#定义KMeans类
class KMeans:
    def __init__(self, data, num_clusters):
        self.data = data   #数值
        self.num_clusters = num_clusters    #簇的个数

    #更新迭代次数
    def train(self, max_iterations):
        #max_iterations 迭代次数
        # 1.先随机选择中心点(质心)，并初始化
        centroids = KMeans.centroids_init(self.data, self.num_clusters)
        # 2.开始训练
        num_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_examples, 1))  # 存放每个数据点对应的类别
        #赋值
        for _ in range(max_iterations):
            # 3.计算样本点到质心的距离，找到最近的
            closest_centroids_ids = KMeans.centroids_find_closest(self.data, centroids)
            # 4.进行中心点位置更新
            centroids = KMeans.centroids_compute(self.data, closest_centroids_ids, self.num_clusters)
        return centroids, closest_centroids_ids

    @staticmethod
    # 在数据中随机找出num_clustres个质心
    def centroids_init(data, num_clustres):
        num_examples = data.shape[0]   #一共数据
        random_ids = np.random.permutation(num_examples)   #随机选择出来id---洗牌
        centroids = data[random_ids[:num_clustres], :]   #随机选择其中几个作为中心点
        return centroids

    @staticmethod
    # 寻找距离数据点最近的质心(欧氏距离)
    def centroids_find_closest(data, centroids):
        num_examples = data.shape[0]
        num_centroids = centroids.shape[0]   #中心点类别
        closest_centroids_ids = np.zeros((num_examples, 1))  # 存放每个数据点对应的距离最近的质心类别
        for example_index in range(num_examples):
            distance = np.zeros((num_centroids, 1))   #初始化距离
            for centroid_index in range(num_centroids):
                distance_diff = data[example_index, :]- centroids[centroid_index, :]  #计算当前中心点和样本的距离
                distance[centroid_index] = np.sum(distance_diff ** 2)   #保存距离
            closest_centroids_ids[example_index] = np.argmin(distance)   #找到最小的距离
        return closest_centroids_ids

    @staticmethod
    # 重新计算簇的质心
    def centroids_compute(data, closest_centroids_ids, num_clustres):
        num_features = data.shape[1]  #特征个数
        centroids = np.zeros((num_clustres, num_features))  # 计算每个特征的均值
        #遍历每个样本点的归属
        for centroid_id in range(num_clustres):
            closest_ids = closest_centroids_ids == centroid_id   #索引
            centroids[centroid_id] = np.mean(data[closest_ids.flatten(), :], axis=0)   #根据索引找出值并计算出均值
        return centroids