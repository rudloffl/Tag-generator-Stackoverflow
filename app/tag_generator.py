#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:51:28 2018

@author: cricket
"""

import pickle
import pandas as pd
import os

#from CustomNMF import CustomNMF


class TagGenerator:
    def __init__(self, customNMF=None, location=''):
        if customNMF is None:
            self.customNMF = self.backup_loading(location)
        else:
            self.customNMF = customNMF

    def backup_loading(self, location):
        with open(os.path.join(os.getcwd(), location, 'customNMF.pickle'), 'rb') as f:
            customNMF = pickle.load(f)
        return customNMF

    def predicttag(self, title='None', text='None', code='None'):
        datainput={'TText':text, 'TTitle':title, 'TCode':code}
        toreturn = {
                'tags':list(self.customNMF.predict(pd.DataFrame.from_dict(datainput, orient='index').T)[0]),
                'keywords':list(self.customNMF.predict_word(pd.DataFrame.from_dict(datainput, orient='index').T)[0]),
                    }
        return toreturn




if __name__ == "__main__":
    taggenerator = TagGenerator()
    title = 'This is a test to see how it goes'
    text = 'This is an other test to see how it goes python json'
    code = 'None'
    print(taggenerator.predicttag(text=text, title=title))