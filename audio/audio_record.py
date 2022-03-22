#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 21:00:52 2021

@author: omar
"""

import sounddevice as sd
from scipy.io.wavfile import write
from audio.audio_converter import convert_audio
from audio.audio_recognition import speech_rec

def record(audio_needed = False):

    fs = 44100  # Sample rate
    seconds = 6  # Duration of recording
    
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    print("Recording...")
    sd.wait()  # Wait until recording is finished
    print("End of Record Session.")
    
    write("recorded.wav", fs, myrecording)  # Save as WAV file
    
    convert_audio()
    
    result = speech_rec(audio_needed)
    
    return result
