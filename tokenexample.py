import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import string

sentences = [
	'create a folder in main directory, called test1',
	'delete the file, called procimage',
	'move file textx from current folder, to one folder down'
]
print("\n")
print(sentences[0])
print(sentences[1])
print(sentences[2]+"\n")

print("words token")
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
print("\n")

print("sequences")
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences[0])
print(sequences[1])
print(sequences[2])

test_data = [
	'clean folder in the previous command',
	'open file called prep using gedit'
]
print("\n")
print("TOKEN FOR TEST DATA")
print(test_data[0])
print(test_data[1])
test_sequences = tokenizer.texts_to_sequences(test_data)
print(test_sequences[0])
print(test_sequences[1])
print("\n")

stopwords = ["a","in","from","to","the"]
newsentences = ["","",""]
print(stopwords)
print("\n")
for i in range(0,3):
	words = sentences[i].split()
	print(words)
	filtered_sentence = ""
	for word in words:
		if word not in stopwords:
			filtered_sentence = filtered_sentence + word + " "
	newsentences[i] = filtered_sentence
print("\n")
print("NEW SENTENCES")	
print(newsentences[0])
print(newsentences[1])
print(newsentences[2])	
print("\n")

table = str.maketrans('','',string.punctuation)

for i in range(0,3):
	words = sentences[i].split()
	print(words)
	filtered_sentence = ""
	for word in words:
		word = word.translate(table)
		if word not in stopwords:
			filtered_sentence = filtered_sentence + word + " "
	sentences.append(filtered_sentence)
print("NEW222 SENTENCES")	
print(sentences[0])
print(sentences[1])
print(sentences[2])
