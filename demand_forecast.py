import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("需求预测系统 - 基于序列分类的智能预测 (优化版)")
print("=" * 80)

print("\n[Step 1] 加载数据...")
df1 = pd.read_excel('data/附件1-商家历史出货量表.xlsx')
df2 = pd.read_excel('data/附件2-商品信息表.xlsx')

df1['date'] = pd.to_datetime(df1['date'])
df1 = df1.sort_values(['seller_no', 'product_no', 'warehouse_no', 'date'])

print(f"历史数据: {len(df1)} 条")
print(f"时间范围: {df1['date'].min()} 至 {df1['date'].max()}")

print("\n[Step 2] 提取序列特征并进行分类...")

def extract_features(group):
    qty = group['qty'].values
    features = {}
    features['mean'] = np.mean(qty)
    features['std'] = np.std(qty)
    features['cv'] = features['std'] / features['mean'] if features['mean'] > 0 else 0
    features['median'] = np.median(qty)
    features['skewness'] = stats.skew(qty) if len(qty) > 2 else 0
    features['seasonal_strength'] = calculate_seasonal_strength(qty)
    features['trend'] = calculate_trend(qty)
    features['zero_ratio'] = np.sum(qty == 0) / len(qty)
    features['data_points'] = len(qty)
    return pd.Series(features)

def calculate_seasonal_strength(qty):
    try:
        n = len(qty)
        if n < 14:
            return 0
        period = 7
        rolling_mean = pd.Series(qty).rolling(window=7, center=True).mean().dropna()
        if len(rolling_mean) < 7:
            return 0
        fft_vals = np.fft.fft(rolling_mean - np.mean(rolling_mean))
        power = np.abs(fft_vals) ** 2
        total_power = np.sum(power)
        if total_power == 0:
            return 0
        peak_freq_idx = np.argmax(power[1:len(power)//2]) + 1
        seasonal_power = power[peak_freq_idx] + power[-peak_freq_idx] if peak_freq_idx < len(power)//2 else power[peak_freq_idx]
        return seasonal_power / total_power
    except:
        return 0

def calculate_trend(qty):
    try:
        x = np.arange(len(qty))
        slope, _, _, _, _ = stats.linregress(x, qty)
        return slope
    except:
        return 0

features_df = df1.groupby(['seller_no', 'product_no', 'warehouse_no']).apply(extract_features).reset_index()

def classify_series(row):
    cv = row['cv']
    seasonal = row['seasonal_strength']
    mean = row['mean']
    zero_ratio = row['zero_ratio']

    if mean > 100:
        return 'D'
    elif cv > 2.0 and zero_ratio > 0.3:
        return 'E'
    elif zero_ratio > 0.4:
        return 'C'
    elif seasonal > 0.4:
        return 'A'
    else:
        return 'B'

features_df['series_type'] = features_df.apply(classify_series, axis=1)

type_names = {
    'A': '稳定季节型', 'B': '一般波动型', 'C': '间歇型',
    'D': '大宗商品型', 'E': '极端波动型'
}

print("\n序列分类结果:")
type_counts = features_df['series_type'].value_counts()
for t in sorted(type_counts.index):
    print(f"  {t}({type_names[t]}): {type_counts[t]} 条")

print("\n[Step 3] 预测方法...")

def forecast_stable_seasonal(group, horizon=15):
    qty = group['qty'].values
    period = 7
    weekly_pattern = []
    for i in range(period):
        week_vals = [qty[j] for j in range(i, len(qty), period) if j < len(qty)]
        weekly_pattern.append(np.mean(week_vals))
    weekly_pattern = np.array(weekly_pattern)

    if len(qty) >= 14:
        trend_adj = (np.mean(qty[-period:]) - np.mean(qty[-2*period:-period])) / period
    else:
        trend_adj = 0

    forecasts = []
    for h in range(horizon):
        day_idx = h % period
        base = weekly_pattern[day_idx]
        forecasts.append(max(0, base + trend_adj * (h // period)))
    return np.array(forecasts)

def forecast_general(group, horizon=15):
    qty = group['qty'].values
    recent = np.mean(qty[-7:])
    if len(qty) >= 14:
        older = np.mean(qty[-14:-7])
        trend = (recent - older) / 7
    else:
        trend = 0
    forecasts = [max(0, recent + trend * h) for h in range(horizon)]
    return np.array(forecasts)

def forecast_intermittent_croston(group, horizon=15):
    qty = group['qty'].values
    nonzero_idx = np.where(qty > 0)[0]
    if len(nonzero_idx) == 0:
        return np.zeros(horizon)

    intervals = np.diff(nonzero_idx)
    nonzero_vals = qty[nonzero_idx]

    avg_interval = np.mean(intervals) if len(intervals) > 0 else 7
    avg_demand = np.mean(nonzero_vals)

    smoothing = 0.1
    interval_pred = avg_interval
    demand_pred = avg_demand

    for i in range(1, len(nonzero_idx)):
        interval_pred = smoothing * intervals[i-1] + (1 - smoothing) * interval_pred
        demand_pred = smoothing * nonzero_vals[i] + (1 - smoothing) * demand_pred

    forecasts = (demand_pred / interval_pred) * np.ones(horizon)
    return forecasts

def forecast_large_volume(group, horizon=15):
    qty = group['qty'].values
    recent = np.mean(qty[-14:])
    if len(qty) >= 28:
        older = np.mean(qty[-28:-14])
        trend = (recent - older) / 14
    else:
        trend = 0
    forecasts = [max(0, recent + trend * h) for h in range(horizon)]
    return np.array(forecasts)

def forecast_extreme_volatility(group, horizon=15):
    qty = group['qty'].values
    recent_median = np.median(qty[-14:]) if len(qty) >= 14 else np.median(qty)
    demand_prob = np.sum(qty > 0) / len(qty)
    forecasts = demand_prob * recent_median * np.ones(horizon)
    return forecasts

forecast_methods = {
    'A': forecast_stable_seasonal,
    'B': forecast_general,
    'C': forecast_intermittent_croston,
    'D': forecast_large_volume,
    'E': forecast_extreme_volatility
}

print("\n[Step 4] 执行预测...")

sequences = df1.groupby(['seller_no', 'product_no', 'warehouse_no'])
series_type_map = features_df.set_index(['seller_no', 'product_no', 'warehouse_no'])['series_type'].to_dict()

results = []
forecast_start = pd.Timestamp('2023-05-16')
forecast_dates = pd.date_range(start=forecast_start, periods=15, freq='D')

for (seller, product, warehouse), group in sequences:
    group = group.sort_values('date')
    series_type = series_type_map.get((seller, product, warehouse), 'B')
    forecast_func = forecast_methods[series_type]
    forecasts = forecast_func(group, horizon=15)

    for i, fdate in enumerate(forecast_dates):
        results.append({
            'seller_no': seller,
            'product_no': product,
            'warehouse_no': warehouse,
            'date': fdate.strftime('%Y-%m-%d'),
            'forecast_qty': round(forecasts[i], 2),
            'series_type': series_type
        })

results_df = pd.DataFrame(results)
print(f"预测完成，共生成 {len(results_df)} 条预测记录")

print("\n[Step 5] 计算模型性能指标（交叉验证）...")

def smape(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return np.mean(np.abs(actual - predicted) / denominator) * 100

def calculate_mae(actual, predicted):
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))

cv_results = []
cv_horizon = 7

for (seller, product, warehouse), group in sequences:
    group = group.sort_values('date')
    qty = group['qty'].values

    if len(qty) >= cv_horizon + 14:
        train_qty = qty[:-cv_horizon]
        test_qty = qty[-cv_horizon:]

        series_type = series_type_map.get((seller, product, warehouse), 'B')
        forecast_func = forecast_methods[series_type]

        train_group = pd.DataFrame({'qty': train_qty})
        predicted = forecast_func(train_group, horizon=cv_horizon)

        mae = calculate_mae(test_qty, predicted)
        rmse = calculate_rmse(test_qty, predicted)
        smape_val = smape(test_qty, predicted)

        cv_results.append({
            'series_type': series_type,
            'seller': seller,
            'product': product,
            'warehouse': warehouse,
            'MAE': mae,
            'RMSE': rmse,
            'SMAPE': smape_val
        })

cv_df = pd.DataFrame(cv_results)

print("\n模型整体性能指标:")
print(f"  平均MAE: {cv_df['MAE'].mean():.4f}")
print(f"  平均RMSE: {cv_df['RMSE'].mean():.4f}")
print(f"  平均SMAPE: {cv_df['SMAPE'].mean():.2f}%")

print("\n各类型序列的预测性能:")
type_performance = cv_df.groupby('series_type')[['MAE', 'RMSE', 'SMAPE']].mean()
type_performance.columns = ['MAE', 'RMSE', 'SMAPE(%)']
print(type_performance.round(4))

print("\n各类型序列数量:")
print(cv_df['series_type'].value_counts().sort_index())

print("\n[Step 6] 输出预测结果...")

output_df = results_df[['seller_no', 'product_no', 'warehouse_no', 'date', 'forecast_qty']].copy()
output_df.to_excel('result/结果表1-预测结果表.xlsx', index=False)
print(f"预测结果已保存至 result/结果表1-预测结果表.xlsx")

cv_df.to_excel('result/模型评估结果.xlsx', index=False)
print(f"模型评估结果已保存至 result/模型评估结果.xlsx")

print("\n" + "=" * 80)
print("预测完成!")
print("=" * 80)