import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

def find_closest_centroids(x, centroids):
    m = x.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((x[i, :] - centroids[j, :])**2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    return idx

def compute_centroids(x, idx, k):
    m, n = x.shape
    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(x[indices, :], axis=1) /
                           len(indices[0])).ravel()

    return centroids

def run_k_means(X, initial_centroids):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    while True:
        idx = find_closest_centroids(X, centroids)
        if((centroids==compute_centroids(X,idx,k)).all()):
            break
        centroids = compute_centroids(X, idx, k)
    return idx, centroids

def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i, :] = X[idx[i], :]

    return centroids

def mysql_con(key):
    mysql = MySQLdb.connect(host='localhost', port=3306, user='root', passwd='123456', db='electricity')
    sql={}
    sql[0]='SELECT date,sum(ele_kWh) as ele_kWh FROM ele_table group by date;'
    sql[1] = 'SELECT date,sum(ele_kWh) as ele_kWh FROM ele_table where festvial = 0 group by date ;'
    sql[2] = 'SELECT date,sum(ele_kWh) as ele_kWh FROM ele_table group by date where festvial = 1;'
    sql[3] = 'SELECT hours,sum(ele_kWh) as ele_kWh FROM ele_table group by hours;'
    sql[4] = 'SELECT hours,sum(ele_kWh) as ele_kWh FROM ele_table group by hours where festvial = 0;'
    sql[5] = 'SELECT hours,sum(ele_kWh) as ele_kWh FROM ele_table group by hours where festvial = 1;'
    sql[6] = 'SELECT * FROM ele_table;'
    sql[7] = 'SELECT id,ele_kWh FROM ele_table;'
    data = pd.read_sql(sql[key-1], con=mysql)
    # 将日期调整为月份按三十天一个月
    if 'date' in data.columns:
        for i in data.index:
            a = data.loc[i, 'date']
            num1 = a[:a.index('.')]
            num2 = a[a.index('.') + 1:]
            num2 = int(num2) / 30
            num2 = int(num1) + num2
            data.loc[i, 'date'] = num2
    if 'hours' in data.columns:
        # 取每个时间段开始作为标志
        for i in data.index:
            a = data.loc[i, 'hours']
            num = int(a[:a.index(':')])
            data.loc[i, 'hours'] = num
    mysql.close()
    return data

if __name__ == "__main__":

    #读取数据，并输入选项
    data=mysql_con(8)
    X = data.values
    print(data)

    #随机化初始点聚类点（3个）
    initial_centroids = init_centroids(X, 3)

    #获取列名
    x=data.columns[0]
    y=data.columns[1]
    print('x=',x)
    print('y=',y)

    sb.set(context="notebook", style="white")
    sb.lmplot(x=x, y=y, data=data, fit_reg=False)


    idx, centroids = run_k_means(X, initial_centroids)
    print("中心点：")
    print(centroids)

    #找到分类好的点集
    cluster1 = X[np.where(idx == 0)[0], :]
    cluster2 = X[np.where(idx == 1)[0], :]
    cluster3 = X[np.where(idx == 2)[0], :]
    #cluster4 = X[np.where(idx == 3)[0], :]

    #作图
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
    ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='Cluster 2')
    ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='Cluster 3')
    #ax.scatter(cluster4[:, 0], cluster4[:, 1], s=30, color='y', label='Cluster 4')
    ax.legend()

    #输入点集
    print("1=",cluster1)
    print("2=", cluster2)
    print("3=", cluster3)

    df = pd.DataFrame(data)
    df.plot()

    # df1 = pd.DataFrame(data, columns=[x,y])
    # df1.plot.pie(subplots=True)

    plt.show()