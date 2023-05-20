import pandas as pd
from preProcess import Tokenize, taged, frequency
from nltk.probability import FreqDist
from sklearn import metrics

def GetNMostFrequentAll(string, n, type_list):
    freq = FreqDist()
    tokens = taged(Tokenize(string))
    for token in tokens:
        if token[1] in type_list:
            frequency(token[0], freq)
    
    return freq.most_common(n)

def Metrics(actual, pred):
    cf = metrics.confusion_matrix(actual, pred)
    acc = metrics.accuracy_score(actual, pred)
    acc = f'{round(acc*100, 2)}%'
    precs = f'{round(cf[1][1]/(cf[1][1] + cf[0][1])*100, 2)}%'

    tp = f'{round(cf[1][1]/(cf[1].sum())*100, 2)}%'
    fp = f'{round(cf[0][1]/(cf[0].sum())*100, 2)}%'
    spy = f'{round(cf[0][0]/(cf[0][0] + cf[1][1])*100, 2)}%'
    error_rate = f'{round((cf[1][1] + cf[0][0])/ cf.sum()*100, 2)}%'

    return acc, precs, tp, fp, spy, error_rate