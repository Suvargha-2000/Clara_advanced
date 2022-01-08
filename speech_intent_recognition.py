# FOR AI 
import nltk
from nltk import tokenize 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
import json 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.python.keras.utils.layer_utils import print_summary
from nltk.stem.snowball import SnowballStemmer

import statistics
from statistics import mode
 
def most_common(List):
    return(mode(List))
   

stemmer = SnowballStemmer(language='english')
nltk.download('punkt')
words = []
labels = []
docs_x = []
docs_y = []

with open('training_data/training.json') as file:
    data = json.load(file)

for intent in data['intents']:
    for pattern in intent['patterns'] :
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])
bag = []
X_sentence = []
X = []
store = []
Y = []
word_bag = []
highest_len = 0

# At first the X created which contains all the sentences and words 
# bag contains all the different types of words 
# Y is where we convert the words to given numbers  

for sentence in docs_x:
    for words in sentence:
        bag.append(stemmer.stem(words))
        # bag.append(words)
    if len(bag) > highest_len :
        highest_len = len(bag)
    X_sentence.append(bag)
    bag = []
for i in X_sentence:
    for ith in i :
        if ith not in word_bag :
            word_bag.append(ith)
for i in docs_y:
    if i not in bag :
        bag.append(i)
for i in docs_y:
    Y.append(bag.index(i))
for i in X_sentence:
    for ith in i :
        store.append(word_bag.index(ith))
    X.append(store)
    store = []

check = []
for i in X :
    for ith in i :
        check.append(ith)
occurrance = most_common(check)
occurrance = check[occurrance]


for i in range(len(X)) : 
    if len(X[i]) < highest_len:
        for ith in range(len(X[i]) , highest_len):
            X[i].append(occurrance)


# print(X)
# print(len(X))
# print(Y)
# print(len(Y))
# print(len(word_bag))
# print(len(bag))
# print(highest_len)

X = np.asarray(X)
Y = np.asarray(Y)
print(X.shape)

chc = input("Do you want to train new models(Y/N) : ")


if chc == "N" : 
    model1 = tf.keras.models.load_model('saved_model/model1')
    model2 = tf.keras.models.load_model('saved_model/model1')
    model3 = tf.keras.models.load_model('saved_model/model1')
    model4 = tf.keras.models.load_model('saved_model/model1')

elif chc == "Y":

    # first model 
    model1 = Sequential()
    model1.add(layers.Dense(highest_len, activation="relu"))
    model1.add(layers.Dense(30 , activation="relu"))
    model1.add(layers.Dense(20 , activation="relu"))
    model1.add(layers.Dense(20 , activation="relu"))
    model1.add(layers.Dense(30 , activation="relu"))
    model1.add(layers.Dense(20 , activation="relu"))
    model1.add(layers.Dense(len(bag) , activation="softmax"))

    model1.compile(optimizer="adam" , loss="sparse_categorical_crossentropy" , metrics="accuracy")

    model1.fit(X , Y , epochs=900)
    model1.save("saved_model/model1")

    # second model 

    model2 = Sequential()
    model2.add(layers.Dense(highest_len, activation="relu"))
    model2.add(layers.Dense(120 , activation="relu"))
    model2.add(layers.Dense(90 , activation="relu"))
    model2.add(layers.Dense(120 , activation="relu"))
    model2.add(layers.Dense(70 , activation="relu"))
    model2.add(layers.Dense(20 , activation="relu"))
    model2.add(layers.Dense(len(bag) , activation="softmax"))

    model2.compile(optimizer="adam" , loss="sparse_categorical_crossentropy" , metrics="accuracy")

    model2.fit(X , Y , epochs=5000)
    model2.save("saved_model/model2")

    # model Three 

    model3 = Sequential()
    model3.add(layers.Dense(highest_len, activation="relu"))
    model3.add(layers.Dense(70 , activation="relu"))
    model3.add(layers.Dense(50 , activation="relu"))
    model3.add(layers.Dense(60 , activation="relu"))
    model3.add(layers.Dense(len(bag) , activation="softmax"))

    model3.compile(optimizer="adam" , loss="sparse_categorical_crossentropy" , metrics="accuracy")

    model3.fit(X , Y , epochs=3000)
    model3.save("saved_model/model3")

    # fourth model 

    model4 = Sequential()
    model4.add(layers.Dense(highest_len, activation="relu"))
    model4.add(layers.Dense(30 , activation="relu"))
    model4.add(layers.Dense(20 , activation="relu"))
    model4.add(layers.Dense(len(bag) , activation="softmax"))

    model4.compile(optimizer="adam" , loss="sparse_categorical_crossentropy" , metrics="accuracy")

    model4.fit(X , Y , epochs=7000)
    model4.save("saved_model/model4")


while True :
    ask_qstn = str(input("Ask The Question : "))
    ask_qstn =  nltk.word_tokenize(ask_qstn)
    sentences = []
    for i in ask_qstn :
        sentences.append(stemmer.stem(i))
        # sentences.append(i)

    for i in range(len(ask_qstn)) :
        
        if sentences[i] in word_bag :
            sentences[i] = word_bag.index(sentences[i])
        else :
            sentences[i] = occurrance

    qstn = sentences
    for i in range(len(qstn)) : 
        if len(qstn) < highest_len:
            for ith in range(len(qstn) , highest_len):
                qstn.append(occurrance)
    qstn = np.asarray(qstn)
    qstn = np.reshape(qstn , (-1 , highest_len))
    
    prediction1 = model1.predict(qstn)
    prediction2 = model2.predict(qstn)
    prediction3 = model3.predict(qstn)
    prediction4 = model4.predict(qstn)
   
    prediction = []
    for i in range(len(prediction1)) :
        prediction.append(prediction1[i]+prediction2[i]+prediction3[i]+prediction4[i])

    print(bag[np.argmax(prediction)])


