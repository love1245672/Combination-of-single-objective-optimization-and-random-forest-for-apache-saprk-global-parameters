import pandas as pd


data = pd.read_csv('grid_d3.csv')


groups = data.groupby('Type')

# 根据分组创建新的数据集
for name, group in groups:
    # 将每个分组保存为单独的CSV文件
    group.to_csv(f'{name}.csv', index=False)