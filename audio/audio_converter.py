#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 20:01:59 2021

@author: omar
"""
import os
import argparse
from pydub import AudioSegment

def convert_audio():

    formats_to_convert = ['.wav']
    
    for (dirpath, dirnames, filenames) in os.walk("."):
        for filename in filenames:
            if filename.endswith(tuple(formats_to_convert)):
                filepath = dirpath + '/' + filename
                (path, file_extension) = os.path.splitext(filepath)
                file_extension_final = file_extension.replace('.', '')
                track = AudioSegment.from_file(filepath,
                        file_extension_final)
                new_filename = "converted.wav"
                wav_filename = new_filename.replace(file_extension_final, 'wav')
                wav_path = dirpath + '/' + wav_filename
                print('CONVERTING: ' + str(filepath))
                file_handle = track.export(wav_path, format='wav')
                os.remove(filepath)