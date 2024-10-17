import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import string
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

#CLEANING THE SENTENCES
table = str.maketrans('','',string.punctuation)
stopwords = ["a","in","from","to","the"]
corpustoken = [] # FOR TOKEN CREATION
corpus = [] # FOR TRAINING
with open('sentences.txt', 'r') as file:
	for line in file:
		sentence = line.strip()
		#print(sentence)
		sentence = sentence.replace(",","")
		#print(sentence)
		words = sentence.split()
		sx = ""
		for word in words:
			word.translate(table)
			if word not in stopwords:
				sx = sx + word + " "
		#print(sx)	
		corpustoken.append(sx)
		corpus.append(sentence)
print("CORPUS - token words")
print(corpustoken)
print("CORPUS - training")
print(corpus)

#TOKENIZING THE SENTENCES	 
vocab_size = 100
tokenizer = Tokenizer(num_words = vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(corpustoken)
word_index = tokenizer.word_index
print("\n WORDS TOKEN")
print(word_index)
#sentencetoken = tokenizer.texts_to_sequences(corpus)
'''print("NEW SENTENCES TOKEN")	
print(sentencetoken)	
print("\n")
'''
#PREPARING DATA FOR TRAINING
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	print(token_list)
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
'''
print("x")
print(xs)
print("labels")
print(labels)
print("y")
print(ys)
'''
