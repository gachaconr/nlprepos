import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import string
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

stopwords = ["a","in","from","to","the"]

sentences = "create a folder in main directory, called test1\ndelete the file, called procimage\nmove file textx from current folder, to one folder down"
#table = str.maketrans(',',' ', string.punctuation)
#sentences.translate(table) not working
sentences = sentences.replace(",","")
print("\n SENTENCE")
print(sentences+"\n")
corpus = sentences.lower().split("\n")
print("CORPUS")
print(corpus)

# REMOVE PUNTUATION
table = str.maketrans('','',string.punctuation)
words = sentences.split()
print(words)
sx = ""
for word in words:
	word.translate(table)
	if word not in stopwords:
		sx = sx + word + " "
print("CORPUS WITHOUT PUNCTUATION")
print(sx)
	 
vocab_size = 100
print("\n WORDS TOKEN")
tokenizer = Tokenizer(num_words = vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
print(word_index)
sentencetoken = tokenizer.texts_to_sequences(corpus)
print("NEW SENTENCES TOKEN")	
print(sentencetoken[0])
print(sentencetoken[1])
print(sentencetoken[2])	
print("\n")

padded = pad_sequences(sentencetoken, padding='pre')
print("TOKEN PADDED")
print(padded)
print("\n")
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)
print("INPUT SEQUENCES")
print(input_sequences)
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
print("\n PADDED SEQUENCES")
print(input_sequences)

xs, labels = input_sequences[:,:-1], input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)
print("x")
print(xs)
print("labels")
print(labels)
print("y")
print(ys)

