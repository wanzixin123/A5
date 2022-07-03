import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

data = read_csv("D:/电力客户/2015.01.01—2016.08.31中国江苏省扬中市1000多家企业每日用电量数据.csv")
df = pd.DataFrame(data, columns=['user_id', 'record_date', 'power_consumption'])
print(df.head())

# 查看数据量
print("当前数据集含有%s行,%s列" % (df.shape[0], df.shape[1]))

# 查看id，可知不同的用户共有1454户
df['user_id'].value_counts()

# 改变数据类型
df.loc[:, 'power_consumption'] = df.power_consumption.astype(float)
df.loc[:, 'user_id'] = df['user_id'].astype(int)
df.loc[:, 'record_date'] = pd.to_datetime(df.record_date)

# 上述数据的时间仅为一列，无法较好的反映工作日与周末及每月具体日期的区别，因此尝试添加列进一步细化日期
# 添加一代表星期的列，isoweekday会根据日期判定是周几
df.loc[:, 'type_day'] = df.record_date.apply(lambda x: x.isoweekday())

# 添加一代表日期的列，day会根据具体日期判定是几号
df.loc[:, 'day_of_month'] = df.record_date.apply(lambda x: x.day)

# 按照id和日期进行重新排序
df = df.sort_values(['user_id', 'record_date'], ascending=[True, True])
df = df.reset_index(drop=True)
print(df.head())

# 筛选出工作日
df0 = df[df.type_day <= 5]

# 按照日期和时间绘制数据透视表，获得不同时间下的用户用电数据
df0 = pd.pivot_table(data=df, columns=['record_date'], values='power_consumption', index=['user_id'])
df0.head()

# 用户用电曲线
fig, ax = plt.subplots(figsize=(16, 9))
plt.plot(df0.T, alpha=1, lw=1)
plt.ylabel('KW')
plt.show()


class KMeans:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    # 通过指定迭代次数终止训练
    def train(self, max_iterations):
        # 1.先随机选择K个中心点
        centroids = KMeans.centroids_init(self.data, self.num_clusters)
        # 2.开始训练
        num_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_examples, 1))  # 存放每个数据点对应的类别
        for _ in range(max_iterations):
            # 3得到当前每一个样本点到K个中心点的距离，找到最近的
            closest_centroids_ids = KMeans.centroids_find_closest(self.data, centroids)
            # 4.进行中心点位置更新
            centroids = KMeans.centroids_compute(self.data, closest_centroids_ids, self.num_clusters)
        return centroids, closest_centroids_ids

    @staticmethod
    # 随机产生num_clustres个质心
    def centroids_init(data, num_clustres):
        num_examples = data.shape[0]
        random_ids = np.random.permutation(num_examples)
        centroids = data[random_ids[:num_clustres], :]
        return centroids

    @staticmethod
    # 寻找距离数据点最近的质心
    def centroids_find_closest(data, centroids):
        num_examples = data.shape[0]
        num_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((num_examples, 1))  # 存放每个数据点对应的距离最近的质心类别
        for example_index in range(num_examples):
            distance = np.zeros((num_centroids, 1))
            for centroid_index in range(num_centroids):
                distance_diff = data[example_index, :] - centroids[centroid_index, :]
                distance[centroid_index] = np.sum(distance_diff ** 2)
            closest_centroids_ids[example_index] = np.argmin(distance)
        return closest_centroids_ids

    @staticmethod
    # 重新计算簇的质心
    def centroids_compute(data, closest_centroids_ids, num_clustres):
        num_features = data.shape[1]
        centroids = np.zeros((num_clustres, num_features))  # 存放每个簇中，每个特征的均值
        for centroid_id in range(num_clustres):
            closest_ids = closest_centroids_ids == centroid_id
            centroids[centroid_id] = np.mean(data[closest_ids.flatten(), :], axis=0)
        return centroids


# 聚类模型
from sklearn.cluster import KMeans


class EnergyFingerPrints():

    def __init__(self, data):
        # will contain the centroid of each cluster
        self.means = []
        self.data = data

    def elbow_method(self, n_clusters):
        """Performs elbow method for a predefined number
        of clusters.

        Parameters
        ----------
        n_clusters : int
            the number of clusters to perform the elbow method

        Returns
        ---------
        A plot the of elbow method
        """
        fig, ax = plt.subplots(figsize=(16, 9))
        distortions = []

        for i in range(1, n_clusters):
            km = KMeans(n_clusters=i,
                        init='k-means++',  # 初始中心簇的获取方式，k-means++一种比较快的收敛的方法
                        n_init=10,  # 初始中心簇的迭代次数
                        max_iter=300,  # 数据分类的迭代次数
                        random_state=0)  # 初始化中心簇的方式
            km.fit(self.data)
            distortions.append(km.inertia_)  # inertia计算样本点到最近的中心点的距离之和

        plt.plot(range(1, n_clusters), distortions, marker='o', lw=1)
        plt.xlabel('聚类数量')
        plt.ylabel('至中心点距离之和')
        plt.show()

    def get_cluster_counts(self):
        return pd.Series(self.predictions).value_counts()

    def labels(self, n_clusters):
        self.n_clusters = n_clusters
        return KMeans(self.n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0).fit(self.data).labels_

    def fit(self, n_clusters):
        """Performs K-means clustering for the load-profiles

        Parameters
        ----------
        n_clusters : int

        Returns
        --------
        count_dict : dict
            The number of load-profiles in each cluster
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(self.n_clusters)
        self.predictions = self.kmeans.fit_predict(self.data)

    def plot(self):
        """Plots all loads in each cluster"""

        self.cluster_names = [str(x) for x in range(self.n_clusters)]
        fig, ax = plt.subplots(figsize=(16, 9))
        for i in range(0, self.n_clusters):
            all_data = []
            for x, y in zip(self.data, self.predictions):
                if y == i:
                    all_data.append(x)
                    plt.subplot(2,1,i+1)
                    plt.plot(x, alpha=0.06, color="blue", lw=2)
                    plt.xlim(0, 96)
                    plt.title('Cluster%s' % (i + 1))
                    plt.ylabel('用电量/kW')

            all_data_array = np.array(all_data)
            mean = all_data_array.mean(axis=0)
            self.means.append(mean)
            plt.plot(mean, color="black", linewidth=4)

        plt.show()

    def plot_energy_fingerprints(self):
        """Plots the mean of each cluster in single plot"""
        fig, ax = plt.subplots(figsize=(16, 9))

        for i, item in enumerate(self.means):
            plt.plot(item, label="cluster %s" % (str(i + 1)))
            plt.xlim(0, 96)
        plt.ylabel('用电量/kW')
        plt.xticks([0, 20, 40, 60, 80, 100, 120,140],
                   ['2015-1-1', '2015-3-31', '2015-6-30', '2015-9-30', '2015-12-31', '2016-3-31', '2016-6-30',
                    '2016-8-31'], rotation=60)
        plt.grid()
        plt.legend()
        plt.show()


# 显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 聚类簇设计
load_data = np.array(df0, dtype=int)
# 导入数据，生成计算模型
energy_clusters = EnergyFingerPrints(load_data)
# 计算聚类簇的距离
energy_clusters.elbow_method(n_clusters=13)

# 构建模型
energy_clusters.fit(n_clusters=2)

# 模型结果分组
energy_clusters.get_cluster_counts()

group = energy_clusters.labels(n_clusters=2)
num = df0.index
cls = pd.DataFrame(list(num))
cls['cluster'] = list(group)
cls.columns = ['user_id', 'cluster']
# 通过排序可以得到每个类中的用户id
cls = cls.sort_values(by='cluster', ascending=True)
cls.reset_index(drop=True)

# 获得属于第一分类簇的用户ide
np.array(cls.loc[cls.cluster == 0].user_id)

# 获得属于第二分类簇的用户id
np.array(cls.loc[cls.cluster == 1].user_id)

# 各组用电数据曲线对比
energy_clusters.plot()
energy_clusters.plot_energy_fingerprints()