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

import pandas as pd

# df = pd.read_csv('file.csv', names=['Name', 'Year', 'Team', 'Points'])
# print(df)

# df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
#                     'B': ['B0', 'B1', 'B2', 'B3'],
#                     'C': ['C0', 'C1', 'C2', 'C3'],
#                     'D': ['D0', 'D1', 'D2', 'D3']})

# df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
#                     'B': ['B4', 'B5', 'B6', 'B7'],
#                     'C': ['C4', 'C5', 'C6', 'C7'],
#                     'D': ['D4', 'D5', 'D6', 'D7']})

# ddf = pd.concat([df1, df2])
# print(ddf)

# data = {'Name': ['Michael Jordan', 'Kobe Bryant', 'Kevin Durant', 'LeBron James', 'Stephen Curry', 'James Harden'],
#         'Year': [1996, 2006, 2014, 2020, 2010, 2009],
#         'Team': ['Chicago Bulls', 'Los Angeles Lakers', 'Oklahoma City Thunder', 'Los Angeles Lakers', 'Golden State Warriors', 'Houston Rockets'],
#         'Points': [2491, 2832, 2593, 1698, 2626, 2191]}

# df = pd.DataFrame(data)

# meanPoints = df.groupby(['Team'])['Points'].mean()
# meanDf = pd.DataFrame(meanPoints).reset_index()
# print(meanDf)

# import pandas as pd

# data = {'Name': ['Michael Jordan', 'Kobe Bryant', 'Kevin Durant', 'LeBron James', 'Stephen Curry', 'James Harden'],
#         'Year': [1996, 2006, 2014, 2020, 2010, 2009],
#         'Team': ['Chicago Bulls', 'Los Angeles Lakers', 'Oklahoma City Thunder', 'Los Angeles Lakers', 'Golden State Warriors', 'Houston Rockets'],
#         'Points': [2491, 2832, 2593, 1698, 2626, 2191]}

# df = pd.DataFrame(data)

# df['Name'] = 'Sr.' + df['Name']
# print(df)

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # データをDataFrameに格納する
# data = {'Name': ['Michael Jordan', 'Kobe Bryant', 'Kevin Durant', 'LeBron James', 'Stephen Curry', 'James Harden'],
#         'Year': [1996, 2006, 2014, 2020, 2010, 2009],
#         'Team': ['Chicago Bulls', 'Los Angeles Lakers', 'Oklahoma City Thunder', 'Los Angeles Lakers', 'Golden State Warriors', 'Houston Rockets'],
#         'Points': [2491, 2832, 2593, 1698, 2626, 2191]}
# df = pd.DataFrame(data)

# # Teamごとの年間得点数を計算する
# # team_points = df.groupby('Team')['Points'].sum()

# # ヒストグラムを作成して表示する

# plt.figure()
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set3.colors)

# sns.histplot(data=df, x="Points", hue="Team", bins=30)
# plt.title('Yearly Point Distribution by Teams')
# plt.xlabel('Points')
# plt.ylabel('Counts')
# plt.legend()
# plt.show()

# import pandas as pd

# df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'],
#                     'value': [1, 2, 3, 4]})
# df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'],
#                     'value': [5, 6, 7, 8]})

# merged = pd.merge(df1, df2, on='key')
# print(merged)

# import pandas as pd

# data = {'name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve', 'Fred'],
#         'age': [25, 30, 45, 21, 38, 42],
#         'gender': ['F', 'M', 'M', 'M', 'F', 'M']}
# df = pd.DataFrame(data)

# grouped = df.groupby('gender')['age'].mean()

# print(grouped)

# import pandas as pd

# data = {'name': ['a','b','c'],
#         'age': [10, 14, 23],
#         'gender': ['f', 'm', 'f']}

# df1 = pd.DataFrame(data)

# grouped = df1.groupby('gender')['age'].mean()
# print(grouped)

# import pandas as pd

# data = {'name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve', 'Fred'],
#         'age': [25, 30, 45, 21, 38, 42],
#         'gender': ['F', 'M', 'M', 'M', 'F', 'M']}

# df = pd.DataFrame(data)
# df.set_index(['gender', 'name'], inplace=True)

# print(df.loc[('F', 'Alice')])

# import pandas as pd

# idx = pd.date_range('20210101', periods=5, freq='D')
# data = {'value': [1, 2, 3, 4, 5]}
# df = pd.DataFrame(data, index=idx)

# print(df.resample('W').sum())

# import pandas as pd
# import matplotlib.pyplot as plt

# data = {'v': [1, 2, 3, 4, 5]}
# df = pd.DataFrame(data)
# df.plot()
# plt.show()

# import pandas as pd

# df = pd.DataFrame({'date': ['2022-11-15', '2021-01-05', '2020-10-31', '2023-06-17']})

# # 文字列を指定した区切り文字で分割　　　　　　　　　　　分割した要素を個々の列とする
# df[['Year', 'Month', 'Day']] = df.date.str.split('-', expand=True)
# print(df)

# import pandas as pd

# data = {'player_id': ['A001', 'B002', 'C003', 'A001', 'C003', 'B002'],
#         'score': [15, 20, 12, 18, 25, 22]}

# df = pd.DataFrame(data)

# total_score = df.groupby('player_id').sum()
# print(total_score)

# import pandas as pd

# df = pd.DataFrame({'item': ['apple', 'banana', 'orange'],
#                    'price': [100, 80, 120],
#                    'quantity': [5, 3, 8]})

# df['total_price'] = df['price'] * df['quantity']
# print(df)
