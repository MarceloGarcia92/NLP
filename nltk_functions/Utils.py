import re

from nltk import pos_tag, word_tokenize, CFG, RegexpParser, ngrams, ne_chunk
from nltk.parse.generate import generate
from preProcess import Tokenize

from parametrs import list_Verbs


def GetNGrams(string, n):
    string = string.split(' ')
    return list(ngrams(string, n))

def NounsCount(tokenized):
    return len([token for token in tokenized if token[1] in ['NN', 'NNP', 'NNPS', 'NNS']])

def PronounsCount(tokenized):
    return len([token for token in tokenized if token[1] in ['PRP', 'PRP$']])

def AdjectivesCount(tokenized):
    return len([token for token in tokenized if token[1] in ['JJ', 'JJS', 'JJR']])

def VerbsCount(tokenized):
    return len([token for token in tokenized if token[1] in ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']])

def AdverbsCount(tokenized):
    return len([token for token in tokenized if token[1] in ['RB', 'RBR']])

def GeoPoliticalCount(taged):
    count = int()
    taged_ner = ne_chunk(taged)
    for tag in taged_ner:
        if bool(re.match('\WGPE\W', str(tag))) == True:
            count += 1
    
    return count

def PersonsCount(taged):
    count = int()
    taged_ner = ne_chunk(taged)
    for tag in taged_ner:
        if bool(re.match('\WPERSON\W', str(tag))) == True:
            count += 1
    
    return count


def OrganizationsCount(taged):
    count = int()
    taged_ner = ne_chunk(taged)
    for tag in taged_ner:
        if bool(re.match('\WORGANIZATION\W', str(tag))) == True:
            count += 1
    
    return count

def cfg_parse(string):
    creation = str()
    sent_tk = pos_tag(word_tokenize(string))
    
    for word in sent_tk:
        if word[1] == 'NNP':
            nnp = f"\' {word[0]} \'" 
        if word[1] in list_Verbs:
            v = f"\' {word[0]} \'"
        if word[1] in 'NN':
            nn = f"\' {word[0]} \'"

    try: 
        grammar = CFG.fromstring(f"""
        S -> NP VP
        VP -> V N
        NP -> {nnp}
        V -> {v}
        N -> {nn}
        """)

        for sentence in generate(grammar):
            creation += ''.join(sentence)
    except:
        pass

    
        
    return creation

def cfg_parse_v2(grammar, list_sent, df, type_tag, type_obj):
    for sentiment in list_sent:
        # Filter the dataset by sentiment
        reviews = df['text'][df['airline_sentiment'] == list_sent[0]].reset_index(drop=True)

        # Tokenize and tag each review
        tagged_reviews = [pos_tag(word_tokenize(r)) for r in reviews]

        # Extract noun phrases from each review
        noun_phrases = list()
        for tagged_review in tagged_reviews:
            parser = RegexpParser(grammar)
            tree = parser.parse(tagged_review)
            for subtree in tree.subtrees():
                if subtree.label() == type_tag:
                    noun_phrase = ' '.join([word for word, tag in subtree.leaves()])
                    noun_phrases.append(noun_phrase)

        filename = f'{type_obj} Phrases for {sentiment.capitalize()} Review.txt'
        with open(filename, 'w') as f:
            for np in noun_phrases:
                f.write(np + '\n')


def AllCapitalizedWordsFromText(string):
    return [word for word in Tokenize(string) if word[0].isupper()]

def AllEmailsFromText(string):
    return [word for word in string.split(' ') if bool(re.match(r'[\w.+-]+@[\w-]+\.[\w.-]+', word)) == True]