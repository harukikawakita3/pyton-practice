# -*- coding: utf-8 -*- #
# import pandas as pd
# import numpy as np

# df = pd.DataFrame({
#     '名前': ['太郎', '花子', '次郎'],
#     '年齢': [20, 25, 30],
#     '性別': ['男', '女', '男']
# })
# df_mean = df.groupby('性別')['年齢'].mean()
# print(df_mean)
# print(list)
# age = list['年齢']
# print(age)
# df_list = pd.DataFrame({
#     '名前': ['太郎','花子','次郎', np.nan],
#     '年齢': ['20','25','30', np.nan],
#     '性別': ['男','女','男', np.nan]
# })
# print(df_list.isnull().values.any())

# import pandas as pd
# import numpy as np

# df = pd.DataFrame({
#     '名前': ['太郎', '花子', '次郎'],
#     '年齢': [20, 25, 30],
#     '性別': [1, 2, 1] # 数字に変換する
# })
# grouped = df.groupby('性別')['年齢'].mean()
# print(grouped)

# import pandas as pd
# import matplotlib.pyplot as plt

# data = {
#     '名前': ['太郎', '花子', '次郎'],
#     '年齢': [20, 25, 30],
#     '性別': ['男', '女', '男'],
#     '身長': [170, 160, 180],
#     '体重': [60, 50, 70]
# }

# df = pd.DataFrame(data)

# plt.xlabel('年齢')
# plt.ylabel('身長')
# plt.title('年齢ごとの身長の平均値')
# age_height = df.groupby('年齢')['身長'].mean()
# plt.plot(age_height.index, age_height.values)
# plt.show()

# mean_df = df.groupby('性別')['身長'].mean()
# height_status = df['身長'].describe()
# weight_status = df['体重'].describe()
# print(f'身長\n{height_status}/n')
# print(f'体重\n{weight_status}')

# print(mean_df)

# import matplotlib.pyplot as plt

# # グラフ用のデータを生成
# x = [1, 2, 3, 4, 5]
# y = [1, 4, 9, 16, 25]

# # 線グラフをプロット
# plt.plot(x, y)

# # グラフをカスタマイズ
# plt.title('Simple Line Graph')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')

# # グラフを表示
# plt.show()