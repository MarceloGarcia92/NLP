import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, RegexpParser

#Lemmatize
from nltk.stem import WordNetLemmatizer, PorterStemmer

def one_hot_hash(samples, dimensionality, max_length):
    results = np.zeros((len(samples), max_length, dimensionality))
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = abs(hash(word)) % dimensionality
            results[i, j, index] = 1


def Tokenize(string): 
    return word_tokenize(string)

def PrintSyntaxTree(string, grammar):
    chunk_parser = RegexpParser(grammar)
    tokens = taged(Tokenize(string))
    return chunk_parser.parse(tokens)

def taged(tokens):
    return pos_tag(tokens)

def frequency(word, freq):
    freq[word.lower()] +=1 
    return freq

def Lemmatize(clean_string):
    lem_string = str()
    lem = WordNetLemmatizer()

    for word in clean_string:
        lem_string += f'{lem.lemmatize(word)} '

    return lem_string

def Stemmed(clean_string):
    stm_words = str()
    stm = PorterStemmer()
    for word in clean_string:
        stm_words += f'{stm.stem(word)} '
    
    return stm_words

def RemoveStopWords(tokenized):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokenized if not word.lower() in stop_words]


def TextAfterRemovingPunctuations(string):
    clean_text = str()
    punctation = re.compile(r'[-.?!,:;()|]')
    for word in Tokenize(string):
        word = punctation.sub(r'', word)
        clean_text += f'{word} '
    return clean_text

def TextAfterRemovingDigits(string):
    clean_text = str()
    punctation = re.compile(r'[0-9]')
    for word in Tokenize(string):
        word = punctation.sub(r'', word)
        clean_text += f'{word} '
    return clean_text


def RefineLemmatize(string):
    return Lemmatize(RemoveStopWords(Tokenize(string)))


def encoded(csv_route, label, name_dest):
    df = pd.read_csv(csv_route)
    le = LabelEncoder()
    df[name_dest] = le.fit_transform(df[label])
    try:
        df.drop(columns='Unnamed: 0', inplace=True)
    except:
        pass

    return df, le

def TFIDFVectorization(list_):
    tf_vect = TfidfVectorizer(lowercase=True, stop_words='english')
    tf_matrix = tf_vect.fit_transform(list_)
    return pd.DataFrame(tf_matrix.toarray(), columns=tf_vect.get_feature_names_out())