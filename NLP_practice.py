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

import nltk
nltk.download('averaged_perceptron_tagger')

text = "this is a sample text that we will use to demonstrate the pos tagging functionality of nltk"
tokens = nltk.word_tokenize(text)
tags = nltk.pos_tag(tokens)

for word, tag in tags:
    print(f"{word} : {tag}")