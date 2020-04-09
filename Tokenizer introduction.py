# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:36:22 2020

@author: Jack
"""

import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import json 


sentences = ['i love my dog' , 'I, love my cat', 'You love my dog!', 
             'Whats wrong with my dog then?', 'Hello']

tk = Tokenizer( oov_token = '<OOV>')
tk.fit_on_texts(sentences)
word_index = tk.word_index
sequences = tk.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post', truncating='post')




with open('Sarcasm_Headlines_Dataset.json', 'r') as f:
    datastore = json.load(f)
    
sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
    

tk = Tokenizer( oov_token = '<OOV>')
tk.fit_on_texts(sentences)
word_index2 = tk.word_index
sequences = tk.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post', truncating='post')
print(sentences[4])
print(padded[4])



