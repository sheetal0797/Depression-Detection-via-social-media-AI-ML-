import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import nltk
from textstat.textstat import textstatistics,legacy_round
from nltk.corpus import stopwords
import gensim
import os
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

dataset = pd.read_csv('CLEF2017Dataset/train.csv',nrows=10000)
dataset.fillna(0, inplace = True)

subjects = np.unique(dataset['ID'])

stop_words = set(stopwords.words('english'))


text = dataset['TEXT'].values
label = dataset['label'].tolist()

textData = []
Y = []
for i in range(len(text)):
    if str(text[i]) != '0':
        tokens = nltk.word_tokenize(text[i])
        textData.append(tokens)
        Y.append(label[i])

'''
if os.path.exists('word2vec.txt'):
    model = gensim.models.KeyedVectors.load_word2vec_format('word2vec.txt', binary=True)
else:
    model = gensim.models.Word2Vec(textData, size=100, window=5, min_count=5, workers=4)
    model.wv.save_word2vec_format('word2vec.txt', binary=False)
'''
data = pd.read_csv("word2vec.txt",header=None,sep=" ")
X = data.values
X = X[:,1:X.shape[1]]
Y = np.asarray(Y)
Y = Y[0:X.shape[0]]
Y = to_categorical(Y)
print(X.shape)
print(Y.shape)

X = X.reshape(X.shape[0],X.shape[1],1,1)
cnn_w2v = Sequential()
cnn_w2v.add(Convolution2D(32, 1, 1, input_shape = (X.shape[1],1,1), activation = 'relu'))
cnn_w2v.add(MaxPooling2D(pool_size = (1, 1)))
cnn_w2v.add(Convolution2D(32, 1, 1, activation = 'relu'))
cnn_w2v.add(MaxPooling2D(pool_size = (1, 1)))
cnn_w2v.add(Flatten())
cnn_w2v.add(Dense(output_dim = 256, activation = 'relu'))
cnn_w2v.add(Dense(output_dim = Y.shape[1], activation = 'softmax'))
print(cnn_w2v.summary())
cnn_w2v.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
hist = cnn_w2v.fit(X, Y, batch_size=16, epochs=10, shuffle=True, verbose=2)
cnn_w2v.save_weights('model/cnn_w2v_weights.h5')            
model_json = cnn_w2v.to_json()
with open("model/cnn_w2v_model.json", "w") as json_file:
    json_file.write(model_json)
json_file.close()    
f = open('model/cnn_w2v_history.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()



