#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:51:28 2018

@author: cricket
"""


import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import string

class Text_tokenizer:

    def __init__(self, punctuation = True, stemming=True):
        self.punctuation = punctuation
        self.stemming = stemming

    def tokenize(self, entry):
        #Breaks in sentences
        sentences = sent_tokenize(entry, language='english')

        #Breaks in words
        tokenizedsentences = []
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tokenizedsentences.append(tokens)

        #Stem the sentences
        if self.stemming:
            stemmer = SnowballStemmer("english")
            stemmed = []
            for sentence in tokenizedsentences:
                sentencestemmed = [stemmer.stem(word.lower()) for word in sentence]
                stemmed.append(sentencestemmed)
        else:
            stemmed=tokenizedsentences

        #Cleans the Stop-words
        stopwords = set(nltk.corpus.stopwords.words('english'))
        stopwords.add('...')
        cleanedsw = []
        for sentence in stemmed:
            tokens = [word for word in sentence if (word not in stopwords) and ((word[:-4] not in stopwords) if word.endswith('_NEG') else True)]
            cleanedsw.append(tokens)

        #Clean the punctuation
        if self.punctuation:
            cleanedpunct = []
            for sentence in cleanedsw:
                tokens = [word for word in sentence if (word not in string.punctuation) and ((word[:-4] not in string.punctuation) if word.endswith('_NEG') else True)]
                cleanedpunct.append(tokens)
        else:
            cleanedpunct = cleanedsw

        #Clean the numbers
        cleanednumber = []
        for sentence in cleanedpunct:
            tokens = [word for word in sentence if not word.isdigit()]
            cleanednumber.append(tokens)


        #Sentences assembly
        assembly = []
        for sentence in cleanednumber:
            tokens = [word for word in sentence]
            assembly.extend(sentence)

        return ' '.join(assembly)



if __name__ == "__main__":
    title = 'This is a test to see how it goes'
    texte = 'This is an other test to see how it goes'
    code = 'None'
    textok = Text_tokenizer()

    print(textok.tokenize(title))