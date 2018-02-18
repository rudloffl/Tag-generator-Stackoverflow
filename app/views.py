#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:54:27 2017

@author: cricket
"""

from flask import Flask, render_template, request
from flask import jsonify
from app import app
import os
import pickle
import pandas as pd

from app import tokenizer
from app.tag_generator import TagGenerator
from app.CustomNMF import CustomNMF




def loadpickle(filename = 'customNMF.pickle'):
    with open(os.path.join(os.getcwd(),'app', 'backup', filename), 'rb') as f:
        return pickle.load(f)

def savepickle(obj, filename = 'customNMF.pickle'):
    with open(os.path.join(os.getcwd(),'app', 'backup', filename), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



try:
    customNMF = loadpickle('customNMF.pickle')
except:
    X_train = pd.read_csv(os.path.join(os.getcwd(), 'app', 'backup', 'X_train.csv'))
    params = {'clf_merge': True,
              'clf_ntopics': 3,
              'clf_ntopwords': 5,
              'clf_serie': ['TText', 'TTitle'],
              'nmf_alpha': 0.1875,
              'nmf_l1_ratio': 0.0,
              'nmf_n_components': 100,
              'nmf_random_state': 0,
              'vect_1_max_df': 1.0,
              'vect_1_max_features': 15000,
              'vect_1_min_df': 10,
              'vect_1_ngram_range': (1, 1),
              }
    popularitytags = loadpickle('popularitytags.pickle')
    customNMF = CustomNMF(popularitytags=popularitytags, **params)
    customNMF.fit(X_train)
    savepickle(customNMF, 'customNMF.pickle')

texttokenizer = tokenizer.Text_tokenizer()
taggenerator = TagGenerator(customNMF)

@app.route('/')
def index():
    return render_template('index.html')




@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      results = taggenerator.predicttag(text=texttokenizer.tokenize(result['text']), title=texttokenizer.tokenize(result['title']))
      return jsonify(results=results)
   else:
       return 'Something went wrong :-('