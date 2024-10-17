import os
import openai
from openai import OpenAI
import streamlit as st
import pandas as pd
import re
import time 
import faiss
import pickle
import numpy as np
import PyPDF2
import concurrent.futures
import glob

def read_idx(file):
    return faiss.read_index(file)

def store_idx(idx, filename):
    faiss.write_index(idx, filename)

def read_mdata(file):
    with open(file, 'rb') as f:
        metadata = pickle.load(f)
    return metadata

def store_mdata(metadata, filename):
    with open(filename, 'wb') as f:
        pickle.dump(metadata, f)

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

def get_embeddings_in_parallel(texts, client):
    embeddings = []
    # Use ThreadPoolExecutor to run tasks in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the function to the input texts in parallel
        futures = {executor.submit(get_embedding, text, client): text for text in texts}
        
        # As results complete, gather the embeddings
        for future in concurrent.futures.as_completed(futures):
            embeddings.append(future.result())
    return embeddings

def store_in_faiss(qa_pairs, client):
    # Initialize FAISS index (with dimensionality matching OpenAI embeddings, which is 1536)
    dimension = 1536  # Embedding size for text-embedding-ada-002
    index = faiss.IndexFlatL2(dimension)  # L2 distance metric (Euclidean distance)

    # Prepare embeddings and metadata (Q&A text)
    embeddings = []
    metadata = []

    embeddings = get_embeddings_in_parallel(qa_pairs, client)
    for qa in qa_pairs:
        metadata.append(qa)  # Store the original Q&A pair as metadata

    # Convert to NumPy array
    embeddings_np = np.array(embeddings).astype('float32')

    # Add embeddings to FAISS index
    index.add(embeddings_np)

    return index, metadata

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

def search_across_multiple_indices(query, index_metadata_label_list, client, top_k):
    """
    This function searches across multiple FAISS indices and returns the most related pairs along with their label names.
    
    Parameters:
    - query: The query string to search for.
    - index_metadata_label_list: A list of tuples (index, metadata, label_name).
    - client: The OpenAI client or embedding generation model.
    - top_k: The total number of top results to return across all indices.

    Returns:
    - A list of dictionaries, each containing 'label', 'chunks' (the surrounding Q&A), and 'distance'.
    """
    combined_results = []

    # Iterate through each FAISS index, metadata, and label
    for index, metadata, label in index_metadata_label_list:
        # Perform search in this specific FAISS index
        results = search_in_faiss(query, index, metadata, client, top_k)
        
        # Add label information to each result and combine
        for surrounding_qas, distance in results:
            combined_results.append({
                'label': label,
                'context': surrounding_qas,
                'distance': distance
            })

    # Sort all combined results by distance to get overall closest matches
    combined_results.sort(key=lambda x: x['distance'])

    # Return the top k results
    return combined_results[:top_k]

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
    data = st.sidebar.file_uploader("Choose a file", type = ['pdf'])
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
                        label_message = f"Label: {st.session_state.label_store[message[4:]]}"                        
                        st.markdown(f"<div class='ai-message'>{label_message}</div>", unsafe_allow_html=True)            
            else:
                if message.startswith("User:"):
                    st.markdown(f"<div class='user-message'>{message}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='ai-message'>{message}</div>", unsafe_allow_html=True)
                    with st.expander("Show context and distance: "):
                        distance_message = f"Distance: {st.session_state.distance_store[message[4:]]}"
                        st.markdown(f"<div class='ai-message'>{distance_message}</div>", unsafe_allow_html=True)
                        context_message = f"Context: <br><br>{st.session_state.context_store[message[4:]]}"                        
                        st.markdown(f"<div class='ai-message'>{context_message}</div>", unsafe_allow_html=True) 
                        label_message = f"Label: {st.session_state.label_store[message[4:]]}"                        
                        st.markdown(f"<div class='ai-message'>{label_message}</div>", unsafe_allow_html=True)             
        #st.markdown('</div>', unsafe_allow_html=True)

def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()  # Extract text from the page
    return text

# Function to clean up the text by removing excessive newlines and spaces
def clean_text(text):
        # Step 1: Remove "\n" followed by digits (like "\n21")
    text = re.sub(r'\n\d+', '', text)

    # Step 1: Replace multiple newlines with a space
    text = re.sub(r'\n+', ' ', text)

    # Step 2: Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Function to extract Q&A pairs with exactly 4 spaces after Q and A
def extract_qa_pairs(text):
    # Regex to match Q&A pairs where Q and A are followed by exactly 4 spaces
    qa_pattern = r'\bQ\s{4}(.+?)\s*A\s{4}(.+?)(?=\bQ\b|\Z)'  # Matches Q followed by 4 spaces, then A followed by 4 spaces

    # Find all Q&A pairs using regex
    qa_pairs = re.findall(qa_pattern, text, flags=re.DOTALL)

    # Combine each Q&A pair into a single sentence
    combined_qa = [f"Q: {clean_text(q).strip()} A: {clean_text(a).strip()}" for q, a in qa_pairs]

    return combined_qa