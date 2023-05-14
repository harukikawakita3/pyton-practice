# import spacy
# nlp = spacy.load("en_core_web_sm")

# doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# for ent in doc.ents:
#     print(ent.text, ent.label_)

# import spacy

# number = [8, 3, 9, 2, 5, 1, 4, 7, 6]

# number.sort(reverse=True)
# second = number[1]
# print(second)

# nlp = spacy.load('en_core_web_sm')
# doc = nlp("this is an example of text!")

# print(type(doc))

# token =doc[0]
# print(type(token))

# span = doc[1:4]
# print(type(span))

# stop_words = nlp.Defaults.stop_words

# print(stop_words)

# import nltk
# nltk.download('averaged_perceptron_tagger')

# text = "this is a sample text that we will use to demonstrate the pos tagging functionality of nltk"
# tokens = nltk.word_tokenize(text)
# tags = nltk.pos_tag(tokens)

# for word, tag in tags:
#     print(f"{word} : {tag}")

# import nltk

# from nltk.corpus import gutenberg
# import string

# nltk.download('gutenberg')
# alice_words = gutenberg.words('carroll-alice.txt')
# alice_words = [word.lower() for word in alice_words if word not in string.punctuation] # 追加
# fdist = nltk.FreqDist(alice_words)
# most_common_word = fdist.most_common(1)
# print(most_common_word[0])

# sense_and_sensibility_words = gutenberg.words('austen-sense.txt')
# unique_words = set(sense_and_sensibility_words)
# num_unique_words = len(unique_words)
# print(num_unique_words)

# word_count = 0
# with open('text_file.txt', 'r') as file:
#     for line in file:
#         word_count += line.count('you')
# print(word_count)


# from collections import Counter

# with open('text_file.txt', 'r') as file:
#     words = file.read().split()
#     freq_counter = Counter(words)

# most_common_word = freq_counter.most_common(1)[0][0]
# frequency = freq_counter[most_common_word]
# print(f'The most common word is "{most_common_word}" with {frequency} occurrences.')

# search_string = 'and'
# with open('text_file.txt', 'r') as file:
#     for line in file:
#         if search_string in line:
#             print(line)

import numpy as np

# matrix = np.random.rand(5, 5)
# average = np.mean(matrix)
# max_value = np.max(matrix)
# min_value = np.min(matrix)
# print(average)
# print(max_value)
# print(min_value)

# matrix = np.random.rand(10, 10)
# diagonal_matrix = np.diag(np.diag(matrix))
# print(diagonal_matrix)

# array_1 = np.array([1,2,3,4,5])
# array_2 = np.array([6,7,8,9,0])

# concatenated = np.concatenate([array_1, array_2])
# print(concatenated)

# import pandas as pd

# data = pd.read_csv('point.csv', index_col=0)

# mean = data.mean()
# standard_deviation = data.std()
# median = data.median()
# maximum = data.max()
# minimum = data.min()

# print(f'Mean:\n{mean}')
# print(f'Standard deviation:\n{standard_deviation}')
# print(f'Median:\n{median}')
# print(f'Maximum value:\n{maximum}')
# print(f'Minimum value:\n{minimum}')

# import pandas as pd
# import numpy as np

# df1 = pd.DataFrame(np.random.randn(3, 2))
# df2 = pd.DataFrame(np.random.randn(2, 4))
# matrix_product = df1.dot(df2)
# print(matrix_product)

# import pandas as pd

# data = pd.read_csv('filtered.csv')

# filtered_data = data[(data['Age'] > 25) & (data['Gender'] == 'Female')]

# filtered_data.to_csv('filtered_data.csv', index=False)
# print(filtered_data)