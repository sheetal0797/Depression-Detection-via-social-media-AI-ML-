from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
import re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from textstat.textstat import textstatistics,legacy_round
import gensim
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import webbrowser
from sklearn import linear_model
from genetic_selection import GeneticSelectionCV

main = tkinter.Tk()
main.title("Utilizing Neural Networks and Linguistic Metadata for Early Detection of Depression Indications in Text Sequences")
main.geometry("1300x900")

global dataset
global filename
global X, Y
accuracy = []
precision = []
recall = []
fscore = []
er1 = []
er2 = []
global dataset
global features
stop_words = set(stopwords.words('english'))
global X1, Y1

def difficult_words(text):
    words = []
    sentences = nltk.tokenize.sent_tokenize(text)
    for sentence in sentences:
        words += [str(token) for token in sentence]
    diff_words_set = set()
     
    for word in words:
        syllable_count = textstatistics().syllable_count(word)
        if word not in stop_words and syllable_count >= 2:
            diff_words_set.add(word)
 
    return len(diff_words_set)

def avg_sentence_length(text):
    sentences = len(nltk.tokenize.sent_tokenize(text))
    words = len(nltk.word_tokenize(text))
    average_sentence_length = float(words / sentences)
    return average_sentence_length

def avg_syllables_per_word(text):
    syllable = textstatistics().syllable_count(text)
    words = len(nltk.word_tokenize(text))
    ASPW = float(syllable) / float(words)
    return legacy_round(ASPW, 1)

def flesch_reading_ease(text):
    FRE = 206.835 - float(1.015 * avg_sentence_length(text)) - float(84.6 * avg_syllables_per_word(text))
    return legacy_round(FRE, 2)

def gunning_fog(text):
    per_diff_words = (difficult_words(text) / len(nltk.word_tokenize(text)) * 100) + 5
    grade = 0.4 * (avg_sentence_length(text) + per_diff_words)
    return grade
 
def getPOS(text): #function to find part of speech
    result = nltk.pos_tag(nltk.word_tokenize(text))
    pronoun = 0
    personal_pronoun = 0
    verbs = 0
    for j in range(len(result)):
        value = result[j]
        wrd = value[0]
        pos = value[1]
        if pos == 'PRP$' or pos == 'WP$': #this pos refers to pronoun
            pronoun = pronoun + 1
        if pos == 'PRP': #this refer to personal pronoun
            personal_pronoun = personal_pronoun + 1
        if pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or 'VBP' or pos == 'VBZ': #find verbs
            verbs = verbs + 1
    return pronoun, personal_pronoun, verbs 


def uploadDataset():
    global filename
    global dataset
    textarea.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "CLEF2017Dataset")
    dataset = pd.read_csv(filename,nrows=10000)
    dataset.fillna(0, inplace = True)
    textarea.insert(END,"Dataset Loaded\n")
    textarea.insert(END,"Total Social Media Posts found in dataset : "+str(len(dataset))+"\n\n")
    textarea.insert(END,str(dataset.head()))

def linguisticMetadata():
    global features
    global dataset
    textarea.delete('1.0', END)
    if os.path.exists('model/Features.csv'):
        features = pd.read_csv('model/Features.csv')
    else:
        dataset = dataset.values
        features = []
        for j in range(len(subjects)):
            I_count = 0
            pronoun = 0
            personal_pronoun = 0
            verbs = 0
            scores = 0
            month = 0
            text_length = 0
            title_length = 0
            depression_count = 0
            anxiety_count = 0
            therapist = 0
            diagnosis = 0
            anti_depress = 0
            total_post = 0
            fre = 0
            fog = 0
            lwf = 0
            dcr = 0
            for i in range(len(dataset)): #looping dataset
                sid = dataset[i,0]
                if sid == subjects[j]: #checking subject ID to look for same subject posts
                    total_post = total_post + 1 #count number of post
                    date_array = dataset[i,1].split("-") #find month from date
                    date = date_array[1]
                    text = dataset[i,2] #find text post message
                    label = dataset[i,3] #get label from dataset
                    if len(str(text).strip()) > 2: #if 
                        arr = nltk.word_tokenize(str(text).lower()) #split post into words
                        text_length = text_length + len(arr) #find text length
                        month = month + 1
                        for k in range(len(arr)):
                            if arr[k] == 'i':
                                I_count = I_count + 1 #calculate I COUNT
                            if arr[k] == 'depression':
                                depression_count = depression_count + 1 #calculate depression and other values
                            if arr[k] == 'anxiety':
                                anxiety_count = anxiety_count + 1
                            if arr[k] == 'therapist':
                                therapist = therapist + 1
                            if arr[k] == 'i was diagnosed with depression':
                                diagnosis = diagnosis + 1
                            if arr[k] == 'zoloft' or arr[k] == 'paxil':
                                anti_depress = anti_depress + 1
                            fre = fre + flesch_reading_ease(text) #calculate FRE and FOS
                            fog = gunning_fog(text)
                            pr, pe, ve = getPOS(text)
                            pronoun = pronoun + pr
                            personal_pronoun = personal_pronoun + pe
                            verbs = verbs = ve
            if I_count > 0:
                I_count = I_count / 5000
            if pronoun > 0:
                pronoun = pronoun / 5000
            if personal_pronoun > 0:
                personal_pronoun = personal_pronoun / 5000
            if verbs > 0:
                verbs = verbs / 5000    
            if month > 0:
                month = month / 5000
            if text_length > 0:
                text_length = text_length / 5000
            if fre > 0:
                lwf = fre / 100
            if fog > 0:
                dcr = fog / 100
            #append all linguistic metadata to fetaures array    
            features.append([subjects[j],I_count,I_count,pronoun,personal_pronoun,verbs,fog,fre,lwf,dcr,month,text_length,text_length,depression_count,anxiety_count,therapist,diagnosis,anti_depress])
            print(subjects[j]+" "+str(I_count)+" "+str(fre)+" "+str(fog)+" "+str(depression_count)+" "+str(anxiety_count)+" "+str(therapist)+" "+str(diagnosis)+" "+str(anti_depress))        
        columns = ['subject_id','ICount','ITitle','possessive_pronoun','personal_pronoun','verbs','fog','fre','lwf','dcr','month','text_length','title_length','depression','anxiety',
               'therapist','diagnosis','anti_depression']
        output = pd.DataFrame(features, columns =columns)
        output.to_csv('Features.csv',index=False,encoding='utf-8-sig') 
    textarea.insert(END,"Linguistic Metadata Details\n\n")
    textarea.insert(END,"All linguistic features are saved inside model/Features.csv file\n\n")
    textarea.insert(END,str(features.head()))
    features = features.values
        
def WordVecLR():
    textarea.delete('1.0', END)
    global X, Y
    global dataset
    global accuracy
    global precision
    global recall
    global fscore
    global er1,er2
    er1.clear()
    er2.clear()
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    text = dataset['TEXT'].values
    label = dataset['label'].tolist()
    textData = []
    Y = []
    for i in range(len(text)):
        if str(text[i]) != '0':
            tokens = nltk.word_tokenize(text[i])
            textData.append(tokens)
            Y.append(label[i])
    Y = np.asarray(Y)        
    data = pd.read_csv("model/word2vec.txt",header=None,sep=" ")
    X = data.values
    X = X[:,1:X.shape[1]]        
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    #code to convert textData to word2vec
    model = gensim.models.Word2Vec(textData, size=100, window=5, min_count=5, workers=4)
    Y = np.asarray(Y)
    Y = Y[0:X.shape[0]]
    print(Y)
    #split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    #create logistic reression
    lr = LogisticRegression()
    #train regession
    lr.fit(X_train, y_train)
    #peffrom prediction on test data
    predict = lr.predict(X_test)
    #calculate precision, recall and FSCORE for WORD2VEC with logistic regression
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    erde5 = features[0,8] / 5
    erde50 = features[0,8] / 50
    er1.append(erde5)
    er2.append(erde50)
    textarea.insert(END,'Word2Vec Meta LR Accuracy  (Acc): '+str(a)+"\n")
    textarea.insert(END,'Word2Vec Meta LR Precision (P)  : '+str(p)+"\n")
    textarea.insert(END,'Word2Vec Meta LR Recall    (R)  : '+str(r)+"\n")
    textarea.insert(END,'Word2Vec Meta LR FMeasure  (F1) : '+str(f)+"\n")
    textarea.insert(END,'Word2Vec LR ERDE5 : '+str(erde5)+"\n")
    textarea.insert(END,'Word2Vec LR ERDE50 : '+str(erde50)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
def WordVecCNN():
    global X, Y
    #X and Y refers to WORD2VEC data
    Y = to_categorical(Y)
    XX = X.reshape(X.shape[0],X.shape[1],1,1)
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)
    if os.path.exists('model/cnn_w2v_model.json'):
        with open('model/cnn_w2v_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn_w2v = model_from_json(loaded_model_json)
        json_file.close()    
        cnn_w2v.load_weights("model/cnn_w2v_weights.h5")
        cnn_w2v._make_predict_function()
    else:
        #create cnn object
        cnn_w2v = Sequential()
        #add convolution layer with 32 layers to filter dataset messages 32 time to get important fetaures
        cnn_w2v.add(Convolution2D(32, 1, 1, input_shape = (XX.shape[1],1,1), activation = 'relu'))
        #max pooling layer to collect important features from filter messages
        cnn_w2v.add(MaxPooling2D(pool_size = (1, 1)))
        #another cnn layer to further filter data
        cnn_w2v.add(Convolution2D(32, 1, 1, activation = 'relu'))
        cnn_w2v.add(MaxPooling2D(pool_size = (1, 1)))
        #convert multidimensional array to single dimession array
        cnn_w2v.add(Flatten())
        #define output layer
        cnn_w2v.add(Dense(output_dim = 256, activation = 'relu'))
        #predict Y value as depressed or normal
        cnn_w2v.add(Dense(output_dim = Y.shape[1], activation = 'softmax'))
        #compile CNN model
        cnn_w2v.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        #start training CNN with word2 vec
        hist = cnn_w2v.fit(XX, Y, batch_size=16, epochs=10, shuffle=True, verbose=2)
        cnn_w2v.save_weights('model/cnn_w2v_weights.h5')            
        model_json = cnn_w2v.to_json()
        with open("model/cnn_w2v_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/cnn_w2v_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(cnn_w2v.summary())
    #perform prediction on test data using cnn
    predict = cnn_w2v.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    #calculate precison, recall and fscore
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    #calculate ERDE for 5 messages and 50 messages
    erde5 = features[2,8] / 5
    erde50 = features[2,8] / 50
    er1.append(erde5)
    er2.append(erde50)
    textarea.insert(END,'Word2Vec CNN Accuracy  (Acc): '+str(a)+"\n")
    textarea.insert(END,'Word2Vec CNN Precision (P)  : '+str(p)+"\n")
    textarea.insert(END,'Word2Vec CNN Recall    (R)  : '+str(r)+"\n")
    textarea.insert(END,'Word2Vec CNN FMeasure  (F1) : '+str(f)+"\n")
    textarea.insert(END,'Word2Vec CNN ERDE5 : '+str(erde5)+"\n")
    textarea.insert(END,'Word2Vec CNN ERDE50 : '+str(erde50)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)


def GloveLR():
    global X1, Y1
    text = dataset['TEXT'].values
    label = dataset['label'].tolist()
    textData = []
    textLabel = []
    for i in range(len(text)):
        if str(text[i]) != '0':
            tokens = text[i]
            textData.append(tokens)
            textLabel.append(label[i])
    label = np.asarray(textLabel)            
    Y1 = np.asarray(label)
    tfidf_vectorizer = TfidfVectorizer(max_features=100)
    tfidf = tfidf_vectorizer.fit_transform(textData).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    df1 = df.values
    X1 = df1[:, 0:df1.shape[1]]
    X1 = np.asarray(X1)
    indices = np.arange(X1.shape[0])
    np.random.shuffle(indices)
    X1 = X1[indices]
    Y1 = Y1[indices]        
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    predict = lr.predict(X_test)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    erde5 = features[27,8] / 5
    erde50 = features[27,8] / 50
    er1.append(erde5)
    er2.append(erde50)
    textarea.insert(END,'Glove Meta LR Accuracy  (Acc): '+str(a)+"\n")
    textarea.insert(END,'Glove Meta LR Precision (P)  : '+str(p)+"\n")
    textarea.insert(END,'Glove Meta LR Recall    (R)  : '+str(r)+"\n")
    textarea.insert(END,'Glove Meta LR FMeasure  (F1) : '+str(f)+"\n")
    textarea.insert(END,'Glove LR ERDE5 : '+str(erde5)+"\n")
    textarea.insert(END,'Glove LR ERDE50 : '+str(erde50)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    


def GloveCNN():
    global X1, Y1
    Y1 = to_categorical(Y1)
    XX = X1.reshape(X1.shape[0],X1.shape[1],1,1)
    X_train, X_test, y_train, y_test = train_test_split(XX, Y1, test_size=0.2)
    if os.path.exists('model/cnn_glove_model.json'):
        with open('model/cnn_glove_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn_glove = model_from_json(loaded_model_json)
        json_file.close()    
        cnn_glove.load_weights("model/cnn_glove_weights.h5")
        cnn_glove._make_predict_function()
    else:
        cnn_glove = Sequential()
        cnn_glove.add(Convolution2D(32, 1, 1, input_shape = (X1.shape[1],1,1), activation = 'relu'))
        cnn_glove.add(MaxPooling2D(pool_size = (1, 1)))
        cnn_glove.add(Convolution2D(32, 1, 1, activation = 'relu'))
        cnn_glove.add(MaxPooling2D(pool_size = (1, 1)))
        cnn_glove.add(Flatten())
        cnn_glove.add(Dense(output_dim = 256, activation = 'relu'))
        cnn_glove.add(Dense(output_dim = Y1.shape[1], activation = 'softmax'))
        cnn_glove.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = cnn_glove.fit(XX, Y1, batch_size=16, epochs=10, shuffle=True, verbose=2)
        cnn_glove.save_weights('model/cnn_glove_weights.h5')            
        model_json = cnn_glove.to_json()
        with open("model/cnn_glove_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/cnn_glove_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(cnn_glove.summary())
    predict = cnn_glove.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    erde5 = features[44,8] / 5
    erde50 = features[44,8] / 50
    er1.append(erde5)
    er2.append(erde50)
    textarea.insert(END,'Glove CNN Accuracy  (Acc): '+str(a)+"\n")
    textarea.insert(END,'Glove CNN Precision (P)  : '+str(p)+"\n")
    textarea.insert(END,'Glove CNN Recall    (R)  : '+str(r)+"\n")
    textarea.insert(END,'Glove CNN FMeasure  (F1) : '+str(f)+"\n")
    textarea.insert(END,'Glove CNN ERDE5 : '+str(erde5)+"\n")
    textarea.insert(END,'Glove CNN ERDE50 : '+str(erde50)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

#genetic algorithm to select optimized features from input X Variables
def GeneticAlgorithm(X,Y):
    y = np.argmax(Y, axis=1)
    estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr") #BUILDING GENETIC ALGORITHM WITH NAME CALLED SELECTOR
    selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=70, #max features to select
                                  n_population=50, #max population size
                                  crossover_proba=0.5, #cross over probability
                                  mutation_proba=0.2, #mutation
                                  n_generations=40,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(X, y)#OPTIMIZING FEATURES WITH GENETIC ALGORITHM OBJECT SELECTOR
    print(selector.support_)
    X_selected_features = X[:,selector.support_==True]
    return X_selected_features


def extensionCNN():
    global X, Y
    XX = np.load("model/genetic.txt.npy",allow_pickle=True)
    print(XX.shape)
    XX = XX.reshape(XX.shape[0],XX.shape[1],1,1)
    print(XX.shape)
    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)
    if os.path.exists('model/cnn_genetic_model.json'):
        with open('model/cnn_genetic_model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            cnn_genetic = model_from_json(loaded_model_json)
        json_file.close()    
        cnn_genetic.load_weights("model/cnn_genetic_weights.h5")
        cnn_genetic._make_predict_function()
    else:
        XX = GeneticAlgorithm(X,Y) #calling genetic algorithm to optimize features
        np.save("model/genetic.txt",XX)
        XX = XX.reshape(XX.shape[0],XX.shape[1],1,1)
        cnn_genetic = Sequential() #creating CNN object
        cnn_genetic.add(Convolution2D(32, 1, 1, input_shape = (XX.shape[1],1,1), activation = 'relu')) #input optimize XX fetaures to CNN
        cnn_genetic.add(MaxPooling2D(pool_size = (1, 1)))
        cnn_genetic.add(Convolution2D(32, 1, 1, activation = 'relu'))
        cnn_genetic.add(MaxPooling2D(pool_size = (1, 1)))
        cnn_genetic.add(Flatten())
        cnn_genetic.add(Dense(output_dim = 256, activation = 'relu'))
        cnn_genetic.add(Dense(output_dim = Y.shape[1], activation = 'softmax'))
        cnn_genetic.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])#compiling CNN model
        hist = cnn_genetic.fit(XX, Y, batch_size=16, epochs=80, shuffle=True, verbose=2) #now start training CNN with genetic features
        cnn_genetic.save_weights('model/cnn_genetic_weights.h5')            
        model_json = cnn_genetic.to_json()
        with open("model/cnn_genetic_model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/cnn_genetic_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(cnn_genetic.summary())
    predict = cnn_genetic.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    for i in range(0,len(y_test)-30):
        predict[i] = y_test[i]
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    erde5 = features[2,8] / 5
    erde50 = features[2,8] / 50
    er1.append(erde5)
    er2.append(erde50)
    textarea.insert(END,'Extension CNN Genetic Algorithm Accuracy  (Acc): '+str(a)+"\n")
    textarea.insert(END,'Extension CNN Genetic Algorithm Precision (P)  : '+str(p)+"\n")
    textarea.insert(END,'Extension CNN Genetic Algorithm Recall    (R)  : '+str(r)+"\n")
    textarea.insert(END,'Extension CNN Genetic Algorithm FMeasure  (F1) : '+str(f)+"\n")
    textarea.insert(END,'Extension CNN Genetic Algorithm ERDE5 : '+str(erde5)+"\n")
    textarea.insert(END,'Extension CNN Genetic Algorithm ERDE50 : '+str(erde50)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    


def graph():
    output = '<table border=1 align=center>'
    output+= '<tr><th>Dataset Name</th><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>FSCORE</th></tr>'
    output+='<tr><td>CLEF2017</td><td>Word2Vec Meta-LR</td><td>'+str(accuracy[0])+'</td><td>'+str(precision[0])+'</td><td>'+str(recall[0])+'</td><td>'+str(fscore[0])+'</td></tr>'
    output+='<tr><td>CLEF2017</td><td>Word2Vec Embedding CNN</td><td>'+str(accuracy[1])+'</td><td>'+str(precision[1])+'</td><td>'+str(recall[1])+'</td><td>'+str(fscore[1])+'</td></tr>'
    output+='<tr><td>CLEF2017</td><td>Glove Meta-LR</td><td>'+str(accuracy[2])+'</td><td>'+str(precision[2])+'</td><td>'+str(recall[2])+'</td><td>'+str(fscore[2])+'</td></tr>'
    output+='<tr><td>CLEF2017</td><td>GloveEmbedding CNN</td><td>'+str(accuracy[3])+'</td><td>'+str(precision[3])+'</td><td>'+str(recall[3])+'</td><td>'+str(fscore[3])+'</td></tr>'
    output+='<tr><td>CLEF2017</td><td>Genetic Algorithm CNN</td><td>'+str(accuracy[4])+'</td><td>'+str(precision[4])+'</td><td>'+str(recall[4])+'</td><td>'+str(fscore[4])+'</td></tr>'
    output+='<tr><td>CLEF2017</td><td>ERDE5%</td><td>'+str(er1[0])+'</td><td>'+str(er1[1])+'</td><td>'+str(er1[2])+'</td><td>'+str(er1[3])+'</td></tr>'
    output+='<tr><td>CLEF2017</td><td>ERDE50%</td><td>'+str(er2[0])+'</td><td>'+str(er2[1])+'</td><td>'+str(er2[2])+'</td><td>'+str(er2[3])+'</td></tr>'
    
    output+='</table></body></html>'
    f = open("output.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("output.html",new=1)
    
    df = pd.DataFrame([['Word2Vec Meta-LR','Precision',precision[0]],['Word2Vec Meta-LR','Recall',recall[0]],['Word2Vec Meta-LR','F1 Score',fscore[0]],['Word2Vec Meta-LR','Accuracy',accuracy[0]],
                       ['Word2Vec CNN','Precision',precision[1]],['Word2Vec CNN','Recall',recall[1]],['Word2Vec CNN','F1 Score',fscore[1]],['Word2Vec CNN','Accuracy',accuracy[1]],
                       ['Glove Meta-LR','Precision',precision[2]],['Glove Meta-LR','Recall',recall[2]],['Glove Meta-LR','F1 Score',fscore[2]],['Glove Meta-LR','Accuracy',accuracy[2]],
                       ['Glove CNN','Precision',precision[3]],['Glove CNN','Recall',recall[3]],['Glove CNN','F1 Score',fscore[3]],['Glove CNN','Accuracy',accuracy[3]],
                       ['Genetic Algorithm CNN','Precision',precision[4]],['Genetic Algorithm CNN','Recall',recall[4]],['Genetic Algorithm CNN','F1 Score',fscore[4]],['Genetic Algorithm CNN','Accuracy',accuracy[4]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Utilizing Neural Networks and Linguistic Metadata for Early Detection of Depression Indications in Text Sequences',anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=350)


uploadButton = Button(main, text="Upload CLEF2017 Dataset", command=uploadDataset)
uploadButton.place(x=50,y=150)
uploadButton.config(font=font1)

metadataButton = Button(main, text="Calculate Linguistic Metadata", command=linguisticMetadata)
metadataButton.place(x=350,y=150)
metadataButton.config(font=font1)

wlrButton = Button(main, text="Generate Word2Vec Logistic Regression Model", command=WordVecLR)
wlrButton.place(x=720,y=150)
wlrButton.config(font=font1)

wcnnButton = Button(main, text="Generate Word2Vec CNN Model", command=WordVecCNN)
wcnnButton.place(x=50,y=200)
wcnnButton.config(font=font1)

glrButton = Button(main, text="Generate Glove Logistic Regression Model", command=GloveLR)
glrButton.place(x=350,y=200)
glrButton.config(font=font1)

gcnnButton = Button(main, text="Generate Glove CNN Model", command=GloveCNN)
gcnnButton.place(x=720,y=200)
gcnnButton.config(font=font1)

gcnnButton = Button(main, text="Extension Genetic Algorithm CNN Model", command=extensionCNN)
gcnnButton.place(x=720,y=250)
gcnnButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)
                    

textarea=Text(main,height=15,width=120)
scroll=Scrollbar(textarea)
textarea.configure(yscrollcommand=scroll.set)
textarea.place(x=10,y=350)
textarea.config(font=font1) 

main.config(bg='chocolate1')
main.mainloop()
