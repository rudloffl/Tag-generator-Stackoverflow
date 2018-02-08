#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:54:27 2017

@author: cricket
"""

from flask import Flask, render_template, request
from app import app



@app.route('/')
def index():
    return render_template('index.html')




@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      print(result)
      return 'Its on the TO-DO list!'
   else:
       return 'go to the main page and complete the form'