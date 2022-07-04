import pandas as pd
import matplotlib.pyplot as plt

# #导入数据
# 省份GDP电量关系
df=pd.read_csv("省份GDP电量关系.csv")
x_axis="近10年年均GDP排行"
y_axis="近10年年均用电量排行"

num_examples=df.shape[0]
x_train=df[[x_axis,y_axis]].values.reshape(num_examples,2)

from sklearn.cluster import KMeans
SSE = []            # 存放每次结果的误差平方和
for k in range(1, 9):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(x_train)
    SSE.append(estimator.inertia_)
X = range(1, 9)

plt.figure(figsize=(15, 10))
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.show()
