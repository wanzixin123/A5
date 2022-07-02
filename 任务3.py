import csv
import statsmodels.api as sm
import pandas as pd
import warnings
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")  # 忽略警告

# 高价值用户的标准：用电量大于平均用电量
#################################### 数据读取与预处理 ################################
data = pd.read_csv('Tianchi_power.csv')
data.isnull().any()  # 没有缺失值
# 一共有1454行，即1454个用户
data.groupby('user_id').count()
# 去除record_date列
data1 = data.drop('record_date', axis=1)
# 所有用户的平均用电量
avg = data1.power_consumption.mean()
data2 = data1.groupby('user_id').mean()
# 删除小于平均用电量的用户
data2 = data2.drop(data2[data2.power_consumption < avg].index)
# 存放所有高价值用户的9月份预测用电量平均值
power_dict = {}
# 存放平稳序列
stable = []
# 存放非平稳序列
unstable = []
##################################### ARIMA模型预测 #####################################
for i in data2.index:
    # 选取user_id=i的用电记录
    data0 = data.loc[data['user_id'] == i]
    # 去除user_id列
    data1 = data0.drop('user_id', axis=1)
    dta = pd.Series(data1["power_consumption"])
    dta.index = pd.Index(data1["record_date"])
    # ARIMA模型
    # 1.寻找差分阶数d
    # 使用ADF检验序列是否平稳
    result = adfuller(dta.values)
    if result[0] < result[4]['1%']:
        stable.append(i)
    else:
        unstable.append(i)
for i in stable:
    data0 = data.loc[data['user_id'] == i]
    # 去除user_id列
    data1 = data0.drop('user_id', axis=1)
    dta = pd.Series(data1["power_consumption"])
    dta.index = pd.Index(data1["record_date"])
    bzs = acorr_ljungbox(dta, lags=24)
    b = bzs['lb_pvalue'].values.tolist()
    if b[0] > 0.05:  # 为白噪声序列，没有预测的意义
        continue
    else:  # 实验发现，没有白噪声数据
        # AIC
        # (p, q) = (sm.tsa.arma_order_select_ic(dta, max_ar=5, max_ma=5, ic='aic')['aic_min_order'])
        # BIC
        (p, q) = (sm.tsa.arma_order_select_ic(dta, max_ar=5, max_ma=5, ic='bic')['bic_min_order'])
        # BIC准则比AIC准则的精度更高
        model = sm.tsa.ARIMA(dta, order=(p, 0, q)).fit()
        # 预测9月份的用电量
        pre1 = model.predict('2016/9/1', '2016/10/1', dynamic=True, typ='levels')
        pre = pd.DataFrame(pre1)
        mean = pre['predicted_mean'].mean()  # 预测值的平均值
        power_dict[i] = mean
        # 利用D - W检验, 检验残差的自相关性
        # 当D - W检验值接近于2时，不存在自相关性，说明模型较好
        resid = model.resid  # 求解模型残差
        print('D-W检验值为{}'.format(durbin_watson(resid.values)))

# 非平稳序列先转换为平稳序列
for i in unstable:
    data0 = data.loc[data['user_id'] == i]
    # 去除user_id列
    data1 = data0.drop('user_id', axis=1)
    dta = pd.Series(data1["power_consumption"])
    dta.index = pd.Index(data1["record_date"])
    diffs = dta.diff(1)
    result = adfuller(diffs.drop(['2015/1/1']).values)  # 一阶差分之后全部变为平稳序列
    if result[0] < result[4]['1%']:
        bzs = acorr_ljungbox(dta, lags=1)
        b = bzs['lb_pvalue'].values.tolist()
        if b[0] < 0.05:
            # AIC
            # (p, q) = (sm.tsa.arma_order_select_ic(dta, max_ar=5, max_ma=5, ic='aic')['aic_min_order'])
            # BIC
            (p, q) = (sm.tsa.arma_order_select_ic(dta, max_ar=5, max_ma=5, ic='bic')['bic_min_order'])
            model = sm.tsa.ARIMA(dta, order=(p, 1, q)).fit()
            # 预测9月份的用电量
            pre1 = model.predict('2016/9/1', '2016/10/1', dynamic=True, typ='levels')
            pre = pd.DataFrame(pre1)
            mean = pre['predicted_mean'].mean()  # 预测值的平均值
            power_dict[i] = mean
            # 利用D - W检验, 检验残差的自相关性
            # 当D - W检验值接近于2时，不存在自相关性，说明模型较好
            resid = model.resid  # 求解模型残差
            print('D-W检验值为{}'.format(durbin_watson(resid.values)))
        else:
            continue
    else:
        print(i)  # 一阶差分之后全都变为平稳序列

####################################### 预测值的处理 #######################################
# 按照预测平均用电量降序排序
power = sorted(power_dict.items(), key=lambda x: x[1], reverse=True)
with open("居民客户的用电缴费习惯分析3.csv", mode='w', encoding='gbk', newline='') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(["用户编号top5", "预测用电量"])
    for i in range(5):
        csvwriter.writerow([power[i][0], power[i][1]])
print("over!!!")