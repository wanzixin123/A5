import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# #导入数据
# 省份GDP电量关系
df=pd.read_csv("省份GDP电量关系.csv")
x_axis="近10年年均GDP排行"
y_axis="近10年年均用电量排行"

#初始图
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.scatter(df[x_axis][:],df[y_axis][:])
plt.title('initial figure')

num_examples=df.shape[0]
x_train=df[[x_axis,y_axis]].values.reshape(num_examples,2)

#指定训练参数
num_clusters=3   #簇的个数
max_iteritions=50   #迭代次数

k_means=joblib.load(filename='k_means---省份GDP电量关系.pkl')
centroids,closest_centroids_ids=k_means.train(max_iteritions)

#joblib.dump(k_means,"k_means---省份GDP电量关系.pkl")

for centroid_id,centroid in enumerate(centroids):
    plt.subplot(1, 3, 2)
    print(centroid_id)
    current_examples_index=(closest_centroids_ids==centroid_id).flatten()
    plt.scatter(df[x_axis][current_examples_index],df[y_axis][current_examples_index],label=centroid_id)
    index=df[x_axis][current_examples_index].index
    print(df.loc[index.values])
    plt.subplot(1,3,3)
    plt.plot(index.values,df[x_axis][current_examples_index])
    plt.title('line chart')

plt.subplot(1, 3, 2)
for centroid_id,centroid in enumerate(centroids):
    plt.scatter(centroid[0],centroid[1],c='black',marker='*')
plt.title('kmeans figure')

plt.legend()
plt.show()
