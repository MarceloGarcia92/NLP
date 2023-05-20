from nltk import word_tokenize
import re

def MakeCorpus(string, review):
    punctation = re.compile(r'[-.?!,:;()|0-9]')
    for list_ in string:
        list_ = punctation.sub('', list_)
        if len(review) == 0:
            review = set(word_tokenize(list_))
        else:
            review_v2 = set(word_tokenize(list_))
            review = review.union(review_v2)

    return review


def CountVectorization(list_str, corpus):
    punctation = re.compile(r'[-.?!,:;()|0-9]')
    list_ = list([0]*len(corpus))
    corpus = list(corpus)
    list_str = punctation.sub('', list_str)
    tokenized = word_tokenize(list_str)
    for token in tokenized:
        list_[list(corpus).index(token)] += 1

    return list_