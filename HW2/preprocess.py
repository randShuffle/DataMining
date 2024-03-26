import pandas as pd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import f_oneway
df = pd.read_csv('./dataset/winequality-white.csv',sep=';')
null_values = df.isnull()
null_values.sum()
print(f'去除重复元素前df的形状: {df.shape}')
df.drop_duplicates(inplace=True)
print(f'去除重复元素后df的形状: {df.shape}')
df['total acidity'] = df['fixed acidity'] + df['volatile acidity']

scaler = MinMaxScaler()

df['quality_normalize'] = scaler.fit_transform(df[['quality']])

df['fixed acidity level'] = pd.qcut(df['fixed acidity'], 3, labels=['low', 'medium', 'high'])

f_values = []
p_values = []

X = df.drop(['quality','fixed acidity level','quality_normalize'], axis=1)  # feature columns
y = df['quality'] 
# 对每个特征进行ANOVA分析
for column in X.columns:
    f_statistic, p_value = f_oneway(X[column], y)
    f_values.append(f_statistic)
    p_values.append(p_value)

anova_results = pd.DataFrame({'Feature': X.columns, 'F Value': f_values, 'P Value': p_values})

anova_results.sort_values(by='F Value', ascending=False, inplace=True)

top_three_features = anova_results.head(3)
print("Top three features with the most significant impact on wine quality:")
print(top_three_features)