import os
from google.protobuf import message
import openai
from openai import OpenAI
import streamlit as st
import pandas as pd
import re
from io import StringIO
from contextlib import redirect_stdout
import time 
import seaborn as sns
import matplotlib.pyplot as plt
import faiss
import pickle
import numpy as np


def get_top_k():
    top_k = st.sidebar.text_input(
        label = 'Enter number of choices you want:',
        value = '',
        placeholder = 'Starting with 2...'
    )
    return top_k 

def get_embedding(text, client, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def search_in_faiss(query, index, metadata, client, top_k):
    # Generate embedding for the query
    query_embedding = get_embedding(query, client)

    # Convert to NumPy array
    query_embedding_np = np.array([query_embedding]).astype('float32')

    # Perform search
    distances, indices = index.search(query_embedding_np, top_k)

    # Retrieve the top-k results (Q&A pairs and distances)
    results = [(i, metadata[i], distances[0][j]) for j, i in enumerate(indices[0])]

    chunks = []
    for idx, result, distance in results:
        # Add the result along with 4 QA pairs before and after
        surrounding_qas = get_surrounding_qas(metadata, idx, context_size=4)
        chunks.append([surrounding_qas, distance])

    return chunks

def get_surrounding_qas(metadata, index, context_size=4):
    # Get 4 QA pairs before and after the current index, ensuring we stay within bounds
    start = max(0, index - context_size)  # Ensure we don't go below index 0
    end = min(len(metadata), index + context_size + 1)  # Ensure we don't exceed the length of the metadata

    # Get the surrounding QA pairs
    surrounding_qas = []
    
    for i in range(start, end):
        if i == index:
            # Highlight the current QA in red
            surrounding_qas.append(f'<span style="color:red;">{metadata[i]}</span>')
        else:
            surrounding_qas.append(metadata[i])

    # Format the surrounding QA pairs as a string
    return "<br>".join(surrounding_qas)

# get personal project key
def set_openai_client(key):

    return OpenAI(api_key=key)

def get_user_key():
    user_key = st.sidebar.text_input(
                label = 'Enter your OpenAI key:',
                value = '',
                placeholder = 'Your key...'
                )
    return user_key

def get_user_data():
    data = st.sidebar.file_uploader("Choose a file", type = ['csv'])
    return data

def get_user_message():
    user_input = st.text_input(
        label="Type your messages here:",
        value="",  # Default value
        placeholder="Type something here...",
    )
    return user_input

def get_openai_response(chunk, prompt, client):
    combined_prompt = f"Here is some information:\n\n{chunk}\n\nBased on the information above, {prompt}"
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": combined_prompt,
            }
        ],
        model="gpt-3.5-turbo",
        )
    return response.choices[0].message.content

def display_message():
    message_container = st.empty()
    with message_container.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            if st.session_state.messages.index(message) < len(st.session_state.messages) - 2:
                if message.startswith("User:"):
                    st.markdown(f"<div class='user-message'>{message}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='ai-message'>{message}</div>", unsafe_allow_html=True)
                    with st.expander("Show context and distance: "):
                        distance_message = f"Distance: {st.session_state.distance_store[message[4:]]}"
                        st.markdown(f"<div class='ai-message'>{distance_message}</div>", unsafe_allow_html=True)
                        context_message = f"Context: <br><br>{st.session_state.context_store[message[4:]]}"                        
                        st.markdown(f"<div class='ai-message'>{context_message}</div>", unsafe_allow_html=True)            
            else:
                time.sleep(2)
                if message.startswith("User:"):
                    st.markdown(f"<div class='user-message'>{message}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='ai-message'>{message}</div>", unsafe_allow_html=True)
                    with st.expander("Show context and distance: "):
                        distance_message = f"Distance: {st.session_state.distance_store[message[4:]]}"
                        st.markdown(f"<div class='ai-message'>{distance_message}</div>", unsafe_allow_html=True)
                        context_message = f"Context: <br><br>{st.session_state.context_store[message[4:]]}"                        
                        st.markdown(f"<div class='ai-message'>{context_message}</div>", unsafe_allow_html=True)            
        #st.markdown('</div>', unsafe_allow_html=True)

