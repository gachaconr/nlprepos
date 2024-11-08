import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import string
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

stopwords = ["a","in","from","to","the","test1","procimage", "textx"]

sentences = "create a folder in main directory, called test1\ndelete the file, called procimage\nmove file textx from current folder, to one folder down"

sentences = sentences.replace(",","")
corpus = sentences.lower().split("\n")
print("CORPUS")
print(corpus[0])
print(corpus[1])
print(corpus[2])

sentences2 = sentences.replace(",","")
data4token = sentences2.lower().split("\n")
# REMOVE PUNTUATION
table = str.maketrans('','',string.punctuation)
for i in range(0,3):
	sx2 = data4token[i]
	words = sx2.split()
	sx = ""
	for word in words:
		word.translate(table)
		if word not in stopwords:
			sx = sx + word + " "
	data4token[i] = sx
print("CORPUS1 WITHOUT STOPWORDS")
print(data4token[0])
print(data4token[1])
print(data4token[2])
	 
vocab_size = 100
print("\n WORDS TOKEN")
tokenizer = Tokenizer(num_words = vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(data4token)
wordindex = tokenizer.word_index
print(wordindex)
sentencetoken = tokenizer.texts_to_sequences(corpus)
print("CORPUS")
print(corpus)
print("NEW SENTENCES TOKEN")
print(sentencetoken)
print("\n")

padded = pad_sequences(sentencetoken, padding='pre')
print("TOKEN PADDED")
print(padded)
print("\n")
input_sequences = []
for line in sentencetoken: 
	for i in range(1, len(line)):
		n_gram_sequence = line[:i+1]
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
'''
embedding_dim = 16
model = tf.keras.models.Sequential([ # MODEL NOT BEEN CREATED PROPERLY, BUG?
	#tf.keras.layers.Embedding(10000, 16), #vocab_size, embedding_dim),
	#tf.keras.layers.GlobalAveragePooling1D(),
	#tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim, return_sequences=True)),
	#tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
	tf.keras.layers.Dense(24, activation='relu'), 
	tf.keras.layers.Dense(1, activation='sigmoid')
])
'''
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
#history = model.fit(xs,ys, epochs=35)

convsize = 3
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (convsize,convsize), activation='relu', input_shape=(113,187,3)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(32, (convsize,convsize), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (convsize,convsize), activation='relu'),  
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (convsize,convsize), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (convsize,convsize), activation='relu'),  
  tf.keras.layers.MaxPooling2D(2,2),  
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00001,l2=0.00001)),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

