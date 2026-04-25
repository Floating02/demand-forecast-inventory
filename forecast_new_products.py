import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("新品预测系统 - 基于分层参考的智能预测")
print("=" * 80)

print("\n[Step 1] 加载数据...")
df1 = pd.read_excel('data/附件1-商家历史出货量表.xlsx')
df2 = pd.read_excel('data/附件2-商品信息表.xlsx')
df3 = pd.read_excel('data/附件3-商家信息表.xlsx')
df4 = pd.read_excel('data/附件4-仓库信息表.xlsx')
df5 = pd.read_excel('data/附件5-新品历史出货量表.xlsx')

df1['date'] = pd.to_datetime(df1['date'])
df5['date'] = pd.to_datetime(df5['date'])

df1 = df1.sort_values(['seller_no', 'product_no', 'warehouse_no', 'date'])
df5 = df5.sort_values(['seller_no', 'product_no', 'warehouse_no', 'date'])

print(f"附件1(历史): {len(df1)} 条, 商家:{df1['seller_no'].nunique()}, 商品:{df1['product_no'].nunique()}")
print(f"附件5(新品): {len(df5)} 条, 商家:{df5['seller_no'].nunique()}, 商品:{df5['product_no'].nunique()}")

print("\n[Step 2] 计算各维度聚合统计特征...")

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

print("计算商家维度统计...")
seller_stats = df1.groupby('seller_no').agg({
    'qty': ['mean', 'std', 'median', lambda x: (x == 0).sum() / len(x)]
}).reset_index()
seller_stats.columns = ['seller_no', 'seller_mean', 'seller_std', 'seller_median', 'seller_zero_ratio']

print("计算仓库维度统计...")
warehouse_stats = df1.groupby('warehouse_no').agg({
    'qty': ['mean', 'std', 'median', lambda x: (x == 0).sum() / len(x)]
}).reset_index()
warehouse_stats.columns = ['warehouse_no', 'warehouse_mean', 'warehouse_std', 'warehouse_median', 'warehouse_zero_ratio']

print("计算商品类别统计...")
product_stats = df1.merge(df2[['product_no', 'category1', 'category2']], on='product_no', how='left')
category_stats = product_stats.groupby('category2').agg({
    'qty': ['mean', 'std', 'median', lambda x: (x == 0).sum() / len(x)]
}).reset_index()
category_stats.columns = ['category2', 'category_mean', 'category_std', 'category_median', 'category_zero_ratio']

print("计算商品维度统计...")
product_agg = product_stats.groupby('product_no').agg({
    'qty': ['mean', 'std', 'median', lambda x: (x == 0).sum() / len(x)]
}).reset_index()
product_agg.columns = ['product_no', 'product_mean', 'product_std', 'product_median', 'product_zero_ratio']

print("\n[Step 3] 定义预测方法...")

def forecast_with_self_data(group, horizon=15):
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
        base = weekly_pattern[day_idx] if weekly_pattern[day_idx] > 0 else np.mean(qty)
        forecasts.append(max(0, base + trend_adj * (h // period)))
    return np.array(forecasts)

def forecast_by_seller_warehouse(seller, warehouse, seller_stats, warehouse_stats, horizon=15):
    seller_row = seller_stats[seller_stats['seller_no'] == seller]
    warehouse_row = warehouse_stats[warehouse_stats['warehouse_no'] == warehouse]

    if len(seller_row) > 0 and len(warehouse_row) > 0:
        base_value = (seller_row['seller_mean'].values[0] + warehouse_row['warehouse_mean'].values[0]) / 2
    elif len(seller_row) > 0:
        base_value = seller_row['seller_mean'].values[0]
    elif len(warehouse_row) > 0:
        base_value = warehouse_row['warehouse_mean'].values[0]
    else:
        base_value = df1['qty'].mean()

    return np.full(horizon, max(0, base_value))

def forecast_by_category(product, seller, warehouse, df2, category_stats, seller_stats, warehouse_stats, horizon=15):
    product_info = df2[df2['product_no'] == product]
    if len(product_info) > 0:
        category = product_info['category2'].values[0]
        cat_row = category_stats[category_stats['category2'] == category]
        if len(cat_row) > 0:
            cat_mean = cat_row['category_mean'].values[0]
        else:
            cat_mean = None
    else:
        cat_mean = None

    seller_row = seller_stats[seller_stats['seller_no'] == seller] if seller in seller_stats['seller_no'].values else None
    warehouse_row = warehouse_stats[warehouse_stats['warehouse_no'] == warehouse] if warehouse in warehouse_stats['warehouse_no'].values else None

    values = []
    if cat_mean is not None:
        values.append(cat_mean)
    if seller_row is not None and len(seller_row) > 0:
        values.append(seller_row['seller_mean'].values[0])
    if warehouse_row is not None and len(warehouse_row) > 0:
        values.append(warehouse_row['warehouse_mean'].values[0])

    if len(values) > 0:
        base_value = np.mean(values)
    else:
        base_value = df1['qty'].mean()

    return np.full(horizon, max(0, base_value))

print("\n[Step 4] 执行预测...")

sequences5 = df5.groupby(['seller_no', 'product_no', 'warehouse_no'])
new_combinations = list(sequences5.groups.keys())
print(f"需要预测的新组合数: {len(new_combinations)}")

results = []
forecast_start = pd.Timestamp('2023-05-16')
forecast_dates = pd.date_range(start=forecast_start, periods=15, freq='D')

reference_stats = {
    'self': 0,
    'seller_warehouse': 0,
    'category': 0,
    'global': 0
}

for idx, (seller, product, warehouse) in enumerate(new_combinations):
    group = sequences5.get_group((seller, product, warehouse)).sort_values('date')
    qty = group['qty'].values
    n_days = len(qty)

    if n_days >= 14:
        forecasts = forecast_with_self_data(group, horizon=15)
        reference_stats['self'] += 1
    else:
        seller_row = seller_stats[seller_stats['seller_no'] == seller]
        warehouse_row = warehouse_stats[warehouse_stats['warehouse_no'] == warehouse]

        if len(seller_row) > 0 or len(warehouse_row) > 0:
            forecasts = forecast_by_seller_warehouse(seller, warehouse, seller_stats, warehouse_stats, horizon=15)
            reference_stats['seller_warehouse'] += 1
        else:
            product_info = df2[df2['product_no'] == product]
            if len(product_info) > 0:
                category = product_info['category2'].values[0]
                cat_row = category_stats[category_stats['category2'] == category]
                if len(cat_row) > 0 and cat_row['category_mean'].values[0] > 0:
                    forecasts = forecast_by_category(product, seller, warehouse, df2, category_stats, seller_stats, warehouse_stats, horizon=15)
                    reference_stats['category'] += 1
                else:
                    base = df1['qty'].mean()
                    forecasts = np.full(15, max(0, base))
                    reference_stats['global'] += 1
            else:
                base = df1['qty'].mean()
                forecasts = np.full(15, max(0, base))
                reference_stats['global'] += 1

    for i, fdate in enumerate(forecast_dates):
        results.append({
            'seller_no': seller,
            'product_no': product,
            'warehouse_no': warehouse,
            'date': fdate.strftime('%Y-%m-%d'),
            'forecast_qty': round(forecasts[i], 2)
        })

    if (idx + 1) % 50 == 0:
        print(f"  已处理: {idx + 1}/{len(new_combinations)}")

results_df = pd.DataFrame(results)
print(f"\n预测完成，共生成 {len(results_df)} 条预测记录")

print("\n预测方法使用统计:")
for method, count in reference_stats.items():
    print(f"  {method}: {count} ({count/len(new_combinations)*100:.1f}%)")

print("\n[Step 5] 保存预测结果...")

results_df.to_excel('result/结果表2-预测结果表.xlsx', index=False)
print(f"预测结果已保存至 result/结果表2-预测结果表.xlsx")

print("\n预测结果预览:")
print(results_df.head(20))

print("\n预测值统计:")
print(f"  总记录数: {len(results_df)}")
print(f"  均值: {results_df['forecast_qty'].mean():.2f}")
print(f"  标准差: {results_df['forecast_qty'].std():.2f}")
print(f"  最小值: {results_df['forecast_qty'].min():.2f}")
print(f"  最大值: {results_df['forecast_qty'].max():.2f}")

print("\n" + "=" * 80)
print("新品预测完成!")
print("=" * 80)