import pandas as pd
import matplotlib.pyplot as plt

# 导入数据
# 峰值分类
df=pd.read_csv("峰值.csv")
x_axis="average"
y_axis="average_norm"

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
