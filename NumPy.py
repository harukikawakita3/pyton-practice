# import numpy as np

# arr = np.array([[1, 2, 3], [4, 5, 6]])
# print(arr)

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# c = a + b
# print(a)
# print(b)
# print(c)

# import numpy as np

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# c = b - a
# print(c)

# import numpy as np

# # # arr = np.array([1,2,3])
# # # print(arr[1])

# arr1 = np.zeros(5)
# print(arr1)

# arr2 = np.zeros((2,3), dtype=int)
# print(arr2)

# arr3 = np.zeros((2,2,3), dtype=np.float64, order='F')
# print(arr3)

# import numpy as np

# a = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])

# b = np.array([[10, 11, 12],[13, 14, 15],[16, 17, 18]])

# print(a * b)

# import numpy as np

# a = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])

# row_0 = a[0]
# col_2 = a[:, 2]

# print(row_0)
# print(col_2)

# import numpy as np

# a = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9],[10, 11, 12],[13, 14, 15],[16, 17, 18]])

# b = a.reshape((3, 6))
# print(b)

# import numpy as np

# a = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
# mean_a = np.mean(a)
# std_a = np.std(a)

# print("Mean:", mean_a)
# print("Std deviantion: ", std_a)

# arr1 = np.array([1, 2, 3])
# print(arr1)
# arr2 = np.zeros((3, 3))
# print(arr2)

# # rrr = np.ones((2, 3))
# # print(rrr)

# arr3 = np.ones((2, 2))
# print(arr3)

# arr4 = np.random.rand(2,3)
# print(arr4)

# arr5 = np.random.randn(2,3)
# print(arr5)

# arr6 = np.array([1, 2, 3])
# arr7 = np.array([4, 5, 6])
# print(arr6 * arr7)

# import numpy as np

# arr1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
# row_2 = arr1[1, :]
# print(row_2)
# center_value = arr1[1, 1]
# print(center_value)
# last_col = arr1[:, 2]
# print(last_col)

# import numpy as np

# arr1 = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
# arr2 = np.array([[9, 8, 7],[6, 5, 4],[3, 2, 1]])

# addition = arr1 + arr2
# print(addition)
# multiplication = arr1 * arr2
# print(multiplication)

# import numpy as np

# # arr = np.arange(27).reshape((3, 3, 3))
# # print(arr)

# # new_arr1 = np.swapaxes(arr, 0, 2)
# # print(new_arr1)

# arr = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])

# # new_arr = np.delete(arr, 1, axis=1)
# # print(new_arr)

# new_arr = arr[np.newaxis, :]
# print(new_arr)

# import numpy as np

# # height x width の2次元配列を生成
# gray = np.array([[1, 2], [3, 4]])

# # 新しい軸を追加して3次元配列に変換
# gray_3d = gray[:, :, np.newaxis]

# print(gray_3d.shape)  # (2, 2, 1)