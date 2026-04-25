import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("促销期间预测系统 - 基于双十一模式的618预测")
print("=" * 80)

print("\n[Step 1] 加载数据...")
df1 = pd.read_excel('data/附件1-商家历史出货量表.xlsx')
df6 = pd.read_excel('data/附件6-促销期间商家出货量表.xlsx')
df2 = pd.read_excel('data/附件2-商品信息表.xlsx')

df1['date'] = pd.to_datetime(df1['date'])
df6['date'] = pd.to_datetime(df6['date'])

df1 = df1.sort_values(['seller_no', 'product_no', 'warehouse_no', 'date'])
df6 = df6.sort_values(['seller_no', 'product_no', 'warehouse_no', 'date'])

print(f"附件1(历史): {len(df1)} 条")
print(f"附件6(双十一促销): {len(df6)} 条")

print("\n[Step 2] 计算促销放大系数...")

sequences1 = df1.groupby(['seller_no', 'product_no', 'warehouse_no'])
sequences6 = df6.groupby(['seller_no', 'product_no', 'warehouse_no'])

promo_multipliers = []

for (seller, product, warehouse), group6 in sequences6:
    if (seller, product, warehouse) in sequences1.groups:
        group1 = sequences1.get_group((seller, product, warehouse)).sort_values('date')
        qty_normal = group1['qty'].values
        qty_promo = group6['qty'].values

        normal_mean = np.mean(qty_normal)
        promo_mean = np.mean(qty_promo)

        if normal_mean > 0:
            multiplier = promo_mean / normal_mean
        else:
            multiplier = 1.0

        promo_multipliers.append({
            'seller_no': seller,
            'product_no': product,
            'warehouse_no': warehouse,
            'normal_mean': normal_mean,
            'promo_mean': promo_mean,
            'multiplier': multiplier
        })

promo_df = pd.DataFrame(promo_multipliers)
print(f"计算的促销系数组合数: {len(promo_df)}")
print("\n促销放大系数统计:")
print(promo_df['multiplier'].describe())

print("\n促销系数分布:")
bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 100]
labels = ['<0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '2.0-3.0', '3.0-5.0', '>5.0']
promo_df['multiplier_bin'] = pd.cut(promo_df['multiplier'], bins=bins, labels=labels)
print(promo_df['multiplier_bin'].value_counts().sort_index())

print("\n[Step 3] 分析双十一日级模式...")

df6['day'] = df6['date'].dt.day
df6['dayofweek'] = df6['date'].dt.dayofweek

daily_pattern = df6.groupby('day')['qty'].mean()
print("\n双十一期间日均需求模式:")
print(daily_pattern)

peak_day = daily_pattern.idxmax()
peak_ratio = daily_pattern.max() / daily_pattern.mean()
print(f"\n峰值日: 11月{peak_day}日, 峰值/均值比: {peak_ratio:.2f}")

print("\n[Step 4] 计算618与双十一的差异调整...")

df1['month'] = df1['date'].dt.month
df1['day'] = df1['date'].dt.day

nonpromo_nov = df1[(df1['month'] == 11) & (df1['day'] > 0)]
nonpromo_jun = df1[df1['month'] == 6]

print(f"11月非促销数据量: {len(nonpromo_nov)}")
print(f"6月数据量: {len(nonpromo_jun)}")

jun_nov_ratio = 1.0
if len(nonpromo_nov) > 0 and len(nonpromo_jun) > 0:
    jun_nov_ratio = nonpromo_jun['qty'].mean() / nonpromo_nov['qty'].mean()
    print(f"6月/11月需求比: {jun_nov_ratio:.4f}")

print("\n[Step 5] 执行618促销预测...")

forecast_start = pd.Timestamp('2023-06-01')
forecast_dates = pd.date_range(start=forecast_start, periods=20, freq='D')

print(f"预测日期范围: {forecast_start} 至 {forecast_dates[-1]}")
print(f"预测天数: {len(forecast_dates)}")

results = []

for (seller, product, warehouse), group6 in sequences6:
    if (seller, product, warehouse) not in sequences1.groups:
        continue

    group1 = sequences1.get_group((seller, product, warehouse)).sort_values('date')
    qty_normal = group1['qty'].values

    normal_mean = np.mean(qty_normal)
    recent_week_mean = np.mean(qty_normal[-7:]) if len(qty_normal) >= 7 else normal_mean

    promo_row = promo_df[(promo_df['seller_no'] == seller) &
                          (promo_df['product_no'] == product) &
                          (promo_df['warehouse_no'] == warehouse)]

    if len(promo_row) > 0:
        multiplier = promo_row['multiplier'].values[0]
    else:
        multiplier = promo_df['multiplier'].median()

    multiplier = np.clip(multiplier, 0.5, 10.0)

    promo_daily = group6.sort_values('date')['qty'].values
    promo_days = len(promo_daily)

    if promo_days >= 11:
        promo_pattern = promo_daily / np.mean(promo_daily)
    else:
        promo_pattern = np.ones(11)

    base_forecast = recent_week_mean * multiplier

    for i, fdate in enumerate(forecast_dates):
        day_of_week = fdate.dayofweek

        if i < len(promo_pattern):
            day_pattern_adj = promo_pattern[i]
        else:
            day_pattern_adj = 1.0

        if day_of_week in [4, 5, 6]:
            weekend_adj = 1.2
        else:
            weekend_adj = 1.0

        forecast_val = base_forecast * day_pattern_adj * weekend_adj / promo_days * 20

        forecast_val = max(0, forecast_val)

        results.append({
            'seller_no': seller,
            'product_no': product,
            'warehouse_no': warehouse,
            'date': fdate.strftime('%Y-%m-%d'),
            'forecast_qty': round(forecast_val, 2)
        })

results_df = pd.DataFrame(results)
print(f"\n预测完成，共生成 {len(results_df)} 条预测记录")

print("\n[Step 6] 预测结果统计...")
print(f"预测组合数: {results_df.groupby(['seller_no', 'product_no', 'warehouse_no']).ngroups}")
print(f"日期范围: {results_df['date'].min()} 至 {results_df['date'].max()}")

print("\n预测值统计:")
print(results_df['forecast_qty'].describe())

print("\n各天预测总量:")
daily_total = results_df.groupby('date')['forecast_qty'].sum()
print(daily_total)

print("\n[Step 7] 保存预测结果...")
results_df.to_excel('result/结果表3-预测结果表.xlsx', index=False)
print("预测结果已保存至 result/结果表3-预测结果表.xlsx")

print("\n" + "=" * 80)
print("促销预测完成!")
print("=" * 80)