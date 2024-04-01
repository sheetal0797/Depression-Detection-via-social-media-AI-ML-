import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import nltk
from textstat.textstat import textstatistics,legacy_round
from nltk.corpus import stopwords

dataset = pd.read_csv('CLEF2017Dataset/train.csv',nrows=10000)
dataset.fillna(0, inplace = True)

subjects = np.unique(dataset['ID'])

stop_words = set(stopwords.words('english'))
dataset = dataset.values

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
 
def getPOS(text):
    result = nltk.pos_tag(nltk.word_tokenize(text))
    pronoun = 0
    personal_pronoun = 0
    verbs = 0
    for j in range(len(result)):
        value = result[j]
        wrd = value[0]
        pos = value[1]
        if pos == 'PRP$' or pos == 'WP$':
            pronoun = pronoun + 1
        if pos == 'PRP':
            personal_pronoun = personal_pronoun + 1
        if pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or 'VBP' or pos == 'VBZ':
            verbs = verbs + 1
    return pronoun, personal_pronoun, verbs           
        
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
    for i in range(len(dataset)):
        sid = dataset[i,0]
        if sid == subjects[j]:
            total_post = total_post + 1
            date_array = dataset[i,1].split("-")
            date = date_array[1]
            text = dataset[i,2]
            label = dataset[i,3]
            if len(str(text).strip()) > 2:
                arr = nltk.word_tokenize(str(text).lower())
                text_length = text_length + len(arr)
                month = month + 1
                for k in range(len(arr)):
                    if arr[k] == 'i':
                        I_count = I_count + 1
                    if arr[k] == 'depression':
                        depression_count = depression_count + 1
                    if arr[k] == 'anxiety':
                        anxiety_count = anxiety_count + 1
                    if arr[k] == 'therapist':
                        therapist = therapist + 1
                    if arr[k] == 'i was diagnosed with depression':
                        diagnosis = diagnosis + 1
                    if arr[k] == 'zoloft' or arr[k] == 'paxil':
                        anti_depress = anti_depress + 1
                    fre = fre + flesch_reading_ease(text)
                    fog = gunning_fog(text)
                    pr, pe, ve = getPOS(text)
                    pronoun = pronoun + pr
                    personal_pronoun = personal_pronoun + pe
                    verbs = verbs = ve
                    
    if 	I_count > 0:
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
    features.append([subjects[j],I_count,I_count,pronoun,personal_pronoun,verbs,fog,fre,lwf,dcr,month,text_length,text_length,depression_count,anxiety_count,therapist,diagnosis,anti_depress])
    print(subjects[j]+" "+str(I_count)+" "+str(fre)+" "+str(fog)+" "+str(depression_count)+" "+str(anxiety_count)+" "+str(therapist)+" "+str(diagnosis)+" "+str(anti_depress))        
                    
columns = ['subject_id','ICount','ITitle','possessive_pronoun','personal_pronoun','verbs','fog','fre','lwf','dcr','month','text_length','title_length','depression','anxiety',
           'therapist','diagnosis','anti_depression']

output = pd.DataFrame(features, columns =columns)
output.to_csv('Features.csv',index=False,encoding='utf-8-sig') 





                    
                    
                
