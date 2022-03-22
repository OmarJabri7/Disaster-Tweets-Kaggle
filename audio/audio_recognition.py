#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 20:01:59 2021

@author: omar
"""

import speech_recognition as sr
import numpy as np
import pandas as pd
import os

def speech_rec(audio_needed = False):

    model = sr.Recognizer()
    
    test = sr.AudioFile('converted.wav')
    
    with test as wav:
        
        audio = model.record(wav)
    
    sentence = model.recognize_google(audio)
    
    print(f"Speech: {sentence}")
    
    sentence_splitted = sentence.split(" ")
    
    if not os.path.exists('data/convos'):
        os.makedirs('data/convos')
    
    with open("data/convos/words.txt", "a") as file_object:
        for word in sentence_splitted:
            file_object.write(word + " ")
        file_object.write("\n")
    if not audio_needed:
        os.remove("converted.wav")
    
    return sentence

