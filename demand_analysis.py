import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("第一步：读取数据")
print("=" * 80)

df1 = pd.read_excel('data/附件1-商家历史出货量表.xlsx')
df2 = pd.read_excel('data/附件2-商品信息表.xlsx')
df3 = pd.read_excel('data/附件3-商家信息表.xlsx')
df4 = pd.read_excel('data/附件4-仓库信息表.xlsx')

df1['date'] = pd.to_datetime(df1['date'])
print(f"附件1数据量: {len(df1)} 条")
print(f"时间范围: {df1['date'].min()} 到 {df1['date'].max()}")
print(f"商家数: {df1['seller_no'].nunique()}")
print(f"商品数: {df1['product_no'].nunique()}")
print(f"仓库数: {df1['warehouse_no'].nunique()}")

print("\n" + "=" * 80)
print("第二步：构建需求量序列")
print("=" * 80)

df1_sorted = df1.sort_values(['seller_no', 'product_no', 'warehouse_no', 'date'])

seq_count = df1_sorted.groupby(['seller_no', 'product_no', 'warehouse_no']).size()
print(f"需求量序列(商家-商品-仓库组合)数量: {len(seq_count)}")
print(f"每条序列平均数据点数: {seq_count.mean():.1f}")

print("\n" + "=" * 80)
print("第三步：提取需求量序列的数理特征")
print("=" * 80)

def extract_features(group):
    qty = group['qty'].values
    features = {}

    features['mean'] = np.mean(qty)
    features['std'] = np.std(qty)
    features['cv'] = features['std'] / features['mean'] if features['mean'] > 0 else 0
    features['median'] = np.median(qty)
    features['min'] = np.min(qty)
    features['max'] = np.max(qty)
    features['range'] = features['max'] - features['min']
    features['q25'] = np.percentile(qty, 25)
    features['q75'] = np.percentile(qty, 75)
    features['iqr'] = features['q75'] - features['q25']
    features['skewness'] = stats.skew(qty) if len(qty) > 2 else 0
    features['kurtosis'] = stats.kurtosis(qty) if len(qty) > 3 else 0

    n = len(qty)
    if n >= 7:
        features['seasonal_strength'] = calculate_seasonal_strength(qty)
    else:
        features['seasonal_strength'] = 0

    if n >= 10:
        features['trend'] = calculate_trend(qty)
    else:
        features['trend'] = 0

    features['zero_ratio'] = np.sum(qty == 0) / n
    features['data_points'] = n

    return pd.Series(features)

def calculate_seasonal_strength(qty):
    try:
        n = len(qty)
        period = 7
        if n < period * 2:
            return 0

        rolling_mean = pd.Series(qty).rolling(window=7, center=True).mean()
        rolling_mean = rolling_mean.dropna()

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
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, qty)
        return slope
    except:
        return 0

print("正在提取各序列的特征，请稍候...")

features_df = df1_sorted.groupby(['seller_no', 'product_no', 'warehouse_no']).apply(extract_features).reset_index()
print(f"特征提取完成，共 {len(features_df)} 条序列")

print("\n特征统计:")
print(features_df.describe().round(4))

print("\n" + "=" * 80)
print("第四步：特征相关性分析")
print("=" * 80)

numeric_cols = ['mean', 'std', 'cv', 'median', 'range', 'iqr', 'skewness', 'kurtosis', 'seasonal_strength', 'trend', 'zero_ratio']
corr_matrix = features_df[numeric_cols].corr()
print("\n特征相关性矩阵:")
print(corr_matrix.round(3))

print("\n" + "=" * 80)
print("第五步：聚类分析")
print("=" * 80)

feature_cols = ['mean', 'std', 'cv', 'skewness', 'kurtosis', 'seasonal_strength', 'trend', 'zero_ratio']
X = features_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n使用肘部法则确定最佳聚类数...")
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    print(f"K={k}, Inertia={kmeans.inertia_:.2f}")

print("\n选择K=5进行聚类分析...")
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
features_df['cluster'] = kmeans.fit_predict(X_scaled)

print("\n各聚类特征统计:")
cluster_summary = features_df.groupby('cluster')[feature_cols].mean()
print(cluster_summary.round(4))

print("\n各聚类样本数量:")
print(features_df['cluster'].value_counts().sort_index())

print("\n" + "=" * 80)
print("第六步：聚类结果解读")
print("=" * 80)

cluster_descriptions = {}
for c in range(n_clusters):
    cluster_data = features_df[features_df['cluster'] == c]
    avg_mean = cluster_data['mean'].mean()
    avg_cv = cluster_data['cv'].mean()
    avg_seasonal = cluster_data['seasonal_strength'].mean()
    avg_trend = cluster_data['trend'].mean()
    avg_zero = cluster_data['zero_ratio'].mean()
    avg_skew = cluster_data['skewness'].mean()

    desc = []
    if avg_zero > 0.3:
        desc.append("高零值率")
    if avg_cv > 1.5:
        desc.append("高波动性")
    elif avg_cv < 0.5:
        desc.append("低波动性")

    if avg_seasonal > 0.15:
        desc.append("强季节性")
    elif avg_seasonal > 0.05:
        desc.append("弱季节性")

    if avg_trend > 0.5:
        desc.append("上升趋势")
    elif avg_trend < -0.5:
        desc.append("下降趋势")

    if avg_skew > 1:
        desc.append("右偏分布")
    elif avg_skew < -1:
        desc.append("左偏分布")

    cluster_descriptions[c] = desc
    print(f"\n聚类 {c}: {', '.join(desc) if desc else '稳定需求'}")
    print(f"  样本数: {len(cluster_data)}, 占比: {len(cluster_data)/len(features_df)*100:.1f}%")
    print(f"  平均需求量: {avg_mean:.2f}, 变异系数: {avg_cv:.2f}, 季节性强度: {avg_seasonal:.3f}")

print("\n" + "=" * 80)
print("第七步：按商品类别分析")
print("=" * 80)

features_df_with_category = features_df.merge(
    df1[['seller_no', 'product_no', 'warehouse_no']].drop_duplicates(),
    on=['seller_no', 'product_no', 'warehouse_no']
)
features_df_with_category = features_df_with_category.merge(
    df2[['product_no', 'category1', 'category2']],
    on='product_no',
    how='left'
)

print("\n各商品一级类别的聚类分布:")
category_cluster = pd.crosstab(features_df_with_category['category1'], features_df_with_category['cluster'])
print(category_cluster)

print("\n各聚类的商品类别偏好:")
for c in range(n_clusters):
    cluster_cat = features_df_with_category[features_df_with_category['cluster'] == c]['category1'].value_counts()
    print(f"\n聚类 {c} 主要商品类别:")
    print(cluster_cat.head(5))

print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)