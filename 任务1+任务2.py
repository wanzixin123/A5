import pandas as pd
import csv

################################## 数据读取与预处理 ################################
df = pd.read_excel('cph.xlsx')
df.columns = ['用户编号', '缴费日期', '缴费金额']
print(df.isnull().any())  # 没有缺失值
# 根据‘用户编号’进行分组
df1 = df.groupby('用户编号')
# 用户总人数
len1 = len(df1)
# 用户总缴费金额
sum1 = df['缴费金额'].sum()
# 平均缴费金额
avg_money = sum1 / len1
# 平均缴费次数
avg_count = df1.count().mean()['缴费日期']
####################################### 任务1 ######################################
with open('居民客户的用电缴费习惯分析1.csv', mode='w', encoding='gbk', newline='') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(["平均缴费次数", "平均缴费金额"])
    csvwriter.writerow([avg_count, avg_money])

# 居民客户的缴费次数
count = df1.count().drop('缴费金额', axis=1)
# 居民客户的缴费金额
money = df1.sum()
# 合并不同用户的缴费次数和缴费金额
df2 = pd.merge(count, money, on='用户编号')
####################################### 任务2 #######################################
with open('居民客户的用电缴费习惯分析2.csv', mode='w', encoding='gbk', newline='') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(["用户编号", "客户类型"])
    for row in df2.itertuples():
        if getattr(row, '缴费日期') > avg_count and getattr(row, '缴费金额') > avg_money:
            csvwriter.writerow([row[0], '高价值型客户'])
        elif getattr(row, '缴费日期') > avg_count and getattr(row, '缴费金额') < avg_money:
            csvwriter.writerow([row[0], '大众型客户'])
        elif getattr(row, '缴费日期') < avg_count and getattr(row, '缴费金额') > avg_money:
            csvwriter.writerow([row[0], '潜力型客户'])
        else:
            csvwriter.writerow([row[0], '低价值型客户'])
