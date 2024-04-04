##-------------------------------------------------------------------------------------##
## Include
##-------------------------------------------------------------------------------------## 
from array import array

from gtts import gTTS
from openai import OpenAI

import os 
from pathlib import Path

import time
import datetime
import base64

import math
import numpy as np
import matplotlib.pyplot as plt
from docx import Document

import streamlit as st
#from audio_recorder_streamlit import audio_recorder
from streamlit_mic_recorder import mic_recorder
from streamlit.logger import get_logger
from typing import Any
import requests
import pandas as pd
import zipfile
from mutagen.mp3 import MP3

##-------------------------------------------------------------------------------------##
## playAudioEmbedded
##-------------------------------------------------------------------------------------##   
# TODO Change playbackRate
def playAudioEmbedded(filename, background=False):

    try:
        with open(filename, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            # <audio controls autoplay="true"> in case you want to show controls
            md = f"""
                <audio id="audio" controls autoplay="true" style="width: 100%;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(
                md,
                unsafe_allow_html=True,
            )
    except FileNotFoundError:
        print("File not found. Check the path variable and filename")
        return    
   
##-------------------------------------------------------------------------------------##
## speechToTextGoogle
##-------------------------------------------------------------------------------------##
import speech_recognition as sr
def speechToTextGoogle(filename, language="en"):
    
    transcript = ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # Adjust the energy threshold based on the surrounding noise level
        r.adjust_for_ambient_noise(source, duration=0.2)
                
        # Listens for the user's input
        question = r.listen(source)
                
        # Using google to recognize audio
        transcript = r.recognize_google(question, language=language)
        return transcript

##-------------------------------------------------------------------------------------##
## speechToTextOpenAI
##-------------------------------------------------------------------------------------##
def speechToTextOpenAI(filename, language="en"):
    client = OpenAI()
    audio_file= open(filename, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        language=language,
        file=audio_file,
        response_format="text"
    )
    
    # Controls
    if transcript.find("Amara.org") > 0: transcript = " "
    if transcript.find("prossimo episodio") > 0: transcript = " "

    return transcript

##-------------------------------------------------------------------------------------##
## chatBot
##-------------------------------------------------------------------------------------##
def chatBot(input, language = "en"):

    prompt = "You are an helpful virtual assistant. Answer with max 50 words"
    if ( language == "it" ): prompt = "Sei un assistente virtuale. Rispondi al massimo con 50 parole"
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": prompt },
        #{"role": "system", "content": "You are the best children story author in the world. Write a story about this topic"},
        {"role": "user", "content": input}
        ]
    )
    return(completion.choices[0].message.content)

##-------------------------------------------------------------------------------------##
## summarizeBot
##-------------------------------------------------------------------------------------##
def summarizeBot(input, language = "en"):

    prompt = "Summarize the provided text"
    if ( language == "it" ): prompt = "Riassumi il testo fornito"
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": input}
        ]
    )
    return(completion.choices[0].message.content)

##-------------------------------------------------------------------------------------##
## sentimentAnalysis
##-------------------------------------------------------------------------------------##
def sentimentAnalysis(transcript):
    content = f"What emotion is the following text expressing?\n{transcript}"
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are a helpful assistant. Answer with one word choosing between Positive, Neutral, Negative"},
        {"role": "user", "content": content}
        ]
    )
    return(completion.choices[0].message.content)

##-------------------------------------------------------------------------------------##
## audioDuration
##-------------------------------------------------------------------------------------##
def audioDuration(filename):
    
    audio = MP3(filename)
    length = audio.info.length
    return length

##-------------------------------------------------------------------------------------##
## streamData
##-------------------------------------------------------------------------------------##
def streamData(input_text, length):
    words = input_text.split()
    nwords = len(words)
    for word in words:
        yield word + " "
        time.sleep(float(length)/nwords)

##-------------------------------------------------------------------------------------##
## textToSpeechGoogle
##-------------------------------------------------------------------------------------##
def textToSpeechGoogle(text, filename, language="en"):

    # Check audio file is not open
    while True:
        try:
            myfile = open(filename, "wb")
            break                             
        except IOError:
            #stopAudio( )
            return

    speech = gTTS(text, lang=language)
    speech.save(filename)
    return audioDuration(filename)

##-------------------------------------------------------------------------------------##
## textToSpeechOpenAI
##-------------------------------------------------------------------------------------##
def textToSpeechOpenAI(text, filename, language="alloy"):
    
    # Check audio file is not open
    while True:
        try:
            myfile = open(filename, "wb")
            break                             
        except IOError:
            #stopAudio( ) 
            return 

    speech_file_path = Path(__file__).parent / filename
    client = OpenAI()
    response = client.audio.speech.create(
        model="tts-1",
        voice=language, # alloy, onyx, fable
        input=text
    )
    response.stream_to_file(speech_file_path)
    return audioDuration(filename)

##-------------------------------------------------------------------------------------##
## createReport
##-------------------------------------------------------------------------------------##
def createReport(language = "en"):
 
    # Read 
    with open('Questions.txt', 'r') as f:
        Questions=f.read()
    n = 0
    Sentiments = [ int(0) ]*1000
    with open('Sentiments.txt', 'r') as f:
        for line in f:
            Sentiments[n] = int(line)
            n += 1
    Sentiments = Sentiments[0:n]

    # Post processing
    Experience = [ int(0) ]*n
    xAxis = [ int(0) ]*n
    Colors = [ 'white' ]*n
    Experience[0] = 0 #Sentiments[0]
    for k in range(1, n):
        Experience[k] = Experience[k-1] + Sentiments[k-1]
        xAxis[k] = k
        if ( Sentiments[k]>0 ): Colors[k] = "green"
        if ( Sentiments[k]==0 ): Colors[k] = "yellow"
        if ( Sentiments[k]<0 ): Colors[k] = "red"

    # Creating the bar plot
    fig = plt.figure()
    yticks = range(math.floor(min(Experience)-1), math.ceil(max(Experience)+1))
    plt.yticks(yticks)
    plt.grid(visible=True, axis='y', linestyle='--')
    #plt.ylim(-1.1, 1.1)
    plt.bar(xAxis, Sentiments, bottom = Experience, color = Colors)
    plt.xlabel("Interactions")
    plt.ylabel("Sentiment evolution")
    plt.savefig('Experience.png')

    # Summarize
    summary = summarizeBot(Questions, language)

    return summary

##-------------------------------------------------------------------------------------##
## chatVoiceBot
##-------------------------------------------------------------------------------------##
def chatVoiceBot(language = "IT"):

    # Initiatlize environment
    if 'k' not in st.session_state:
        st.session_state['k'] = str(0)
        fq = open('Questions.txt', 'w'); fq.close()
        fs = open('Sentiments.txt', 'w'); fs.close()
    inputFile = "input.wav" 
    outputFile = "output.mp3"

    # Initialize session
    withAudio = st.sidebar.toggle('Audio')
    withMic = st.sidebar.toggle('Mic')
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if ( withAudio and withMic ): playAudioEmbedded(outputFile)

    # Chat prompt
    prompt = ""
    prompt = st.chat_input("Write here...")

    # Audio prompt
    volume = -1
    question = ""
    if ( ( not prompt ) and withMic ): 
        
        audio = mic_recorder(start_prompt="Start recording...", stop_prompt="Stop recording...", key='recorder', just_once=True, use_container_width=True)
        if audio: 
            audio_bytes = audio['bytes']
            volume = len(audio_bytes)
        if ( volume > 0 ):
            with open(inputFile, mode='wb') as f:
                f.write(audio_bytes)
            question=speechToTextOpenAI(inputFile, language)
            
        # #Version listening in the background (valid locally)     
        # with st.spinner('\n Speak up \n'):
        #     volume = recordAudio(inputFile)
        #     if ( volume > 0 ): question=speechToTextOpenAI(inputFile, language)
        #     else: 
        #         time.sleep(1) 
        #         st.rerun()
    if ( len(question) > 0 ): prompt = question 

    if prompt:

        # Chat message question
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Commands
        question = prompt
        answer = ""
        if ( question == "/summary" ):
            answer = createReport(language)
            answer = "Below a summary of the conversation \n \n \n" + answer
        elif ( question == "/sentiment" ):
            answer = "Below a summary of the sentiment of the conversation \n "
            createReport(language)
        else:
            # Chatbot
            if ( len(question) > 0 ): answer=chatBot(question, language)
    
        # Chat message answer
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
        if ( question == "/sentiment" ): st.image('Experience.png')
        
        # Play audio
        if ( withAudio and len(answer) > 0 ): textToSpeechGoogle(answer, outputFile, language)
        #if ( withAudio and len(answer) > 0 ): textToSpeechOpenAI(answer, outputFile, "onyx")
        #if ( withAudio and len(answer) > 0 and ( not withMic ) ): playAudio(outputFile)
        if ( withAudio and len(answer) > 0 and ( not withMic ) ): playAudioEmbedded(outputFile)

        # Sentiment analysis
        sentiment = ""
        sentiment = sentimentAnalysis(question)
        sentimentId = 0; 
        if sentiment == "Positive": sentimentId = 1
        if sentiment == "Negative": sentimentId = -1

        # Store results
        if ( 'k' in st.session_state ):
            fq = open('Questions.txt', 'a'); fq.write(question + '\n'); fq.close()
            fs = open('Sentiments.txt', 'a'); fs.write(str(sentimentId) + '\n'); fs.close()
            st.session_state['k'] = str(int(st.session_state['k']) + 1)

