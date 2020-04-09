# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 19:46:55 2020

@author: Jack
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.compat.v1.enable_eager_execution()


imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train, test = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s,l in train:
    training_sentences.append(str(s.numpy()))
    training_labels.append(str(l.numpy()))
    
for s,l in test:
    testing_sentences.append(str(s.numpy()))
    testing_labels.append(str(l.numpy()))
    
train_labels_final = np.array(training_labels)
test_labels_final = np.array(testing_labels)

vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = "post"
oov_tok = "<OOV>"

tk = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tk.fit_on_texts(training_sentences)
word_index = tk.word_index
sequences = tk.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

testing_sequence = tk.texts_to_sequences(testing_setences)
testing_padded = pad_sequences(testing_sequence, maxlen=max_length)


model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                       input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(padded, train_labels_final, epochs=5)
pred = model.predict(testing_padded)
final_pred = []
for i in pred:
    if i > 0.5:
        final_pred.append(1)
    else:
        final_pred.append(0)

correct = 0
for i in range(len(final_pred)):
    if final_pred[i] == int(test_labels_final[i]):
        correct = correct + 1
    
accuracy = correct/len(final_pred)

print(f"Accuracy = {accuracy}")
