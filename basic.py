# x = 10
# y = 3
# print(x + y)
# print(x * y)

# name = "Taro"
# age = 25
# print(f"私は{name}です。年齢は{age}です。")

# a = [10, 20, 30, 40, 50]
# print(a[1])

# person = {"name": "Taro", "age": "25", "gender": "male"}
# print(person["name"])

# for i in range(5):
#     print(i)

# i = 10
# while i > 0:
#     print(i)
#     i -= 1

# def reverse_string(s):
#     return s[::-1]
# print(reverse_string("hello"))

# def extract_even_number(lst):
#     new_lst = []
#     for i in lst:
#         if i % 2 == 0:
#             new_lst.append(i)
#     return new_lst

# print(extract_even_number([1,2,3,4,5]))

# class Person:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age

#     def greet(self):
#         return f"私の名前は{self.name}です。年齢は{self.name}です。"

# person = Person("Taro", 25)
# print(person.greet())

# class Rectangle:
#     def __init__(self, width, height):
#         self.width = width
#         self.height = height
#     def area(self):
#         return self.width * self.height

# rectangle = Rectangle(10, 10)
# print(rectangle.area())

# import math

# # 円周率の値を取得
# pi = math.pi
# print(pi)

# # sin関数の計算
# angle = math.pi / 4
# result = math.sin(angle)
# print(result)

# # 平方根の計算
# num = 25
# result = math.sqrt(num)
# print(result)

# # 切り上げ・切り捨て
# x = 3.14159265
# print(math.ceil(x))   # 切り上げ(4)
# print(math.floor(x))  # 切り捨て(3)

# import math
# print(math.pi)

# import os
# files = os.listdir()
# for file in files:
#     print(file)

# with open("basic.py", encoding='utf-8') as f:
#     print(f.read())

# name = input("お名前はなんですか? :")
# print("こんにちは、" + name + "さん!")

# import random

# hands = ['グー', 'チョキ', 'パー']

# while True:
#     computer_hand = random.choice(hands)
#     user_hand = input("グー,チョキ,パー から選んでください: ")
#     print(f"cpuの手は{computer_hand}です")
#     if user_hand == computer_hand:
#         print("あいこです： 次は何を出しますか？: ")
#     elif (user_hand == 'グー' and computer_hand == 'チョキ') or (user_hand == 'チョキ' and computer_hand == 'パー') or (user_hand == 'パー' and computer_hand == 'グー'):
#         print("youの勝ちです")
#     elif (user_hand == 'パー' and computer_hand == 'チョキ') or (user_hand == 'グー' and computer_hand == 'パー') or (user_hand == 'チョキ' and computer_hand == 'グー'):
#         print("cpuの勝ちです")
#     else:
#         print("正しく入力してください")

#     print()

