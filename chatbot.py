import os
import openai
from openai import OpenAI
import streamlit as st
from constants import WRAPPER_PROMPT, MAIN_PAGE_STYLE, INPUT_BOX_STYLE
from chatbot_func import get_user_data, get_user_message, get_user_key, set_openai_client, display_message, get_embedding, search_in_faiss, get_openai_response, get_top_k, store_in_faiss
from chatbot_func import read_idx, store_idx, read_mdata, store_mdata, extract_text_from_pdf, clean_text, extract_qa_pairs, get_embeddings_in_parallel, search_across_multiple_indices
import concurrent.futures
import pandas as pd
import time
import faiss
import pickle
import PyPDF2
import glob

my_key = os.environ.get('MY_API_KEY','')
wrapper_prompt = WRAPPER_PROMPT
main_style = MAIN_PAGE_STYLE
input_box_style = INPUT_BOX_STYLE

def main():
    st.markdown(main_style, unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'index_metadata_label_list' not in st.session_state:
        st.session_state.index_metadata_label_list = []

    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    if 'ai_response' not in st.session_state:
        st.session_state.ai_response = ""

    if "context_store" not in st.session_state:
        st.session_state.context_store = {}

    if "distance_store" not in st.session_state:
        st.session_state.distance_store = {}

    if "label_store" not in st.session_state:
        st.session_state.label_store = {}

    user_key = get_user_key()
    client = set_openai_client(user_key)
    user_data = None
    user_data = get_user_data()
    top_k = get_top_k()

    if st.sidebar.button("Read Data"):
        index_list = glob.glob("index_*.index", recursive=False)
        mdata_list = glob.glob("metadata_*.pkl", recursive=False)
        st.session_state.index_metadata_label_list = []

                # Iterate through the index files and find matching metadata files
        for index_file in index_list:
                    # Extract the label from the index file (e.g., 'index_a.index' -> 'a')
            label = index_file.replace("index_", "").replace(".index", "")
            idx = read_idx(index_file)
                    # Construct the corresponding metadata file name (e.g., 'metadata_a.pkl')
            metadata_file = f"metadata_{label}.pkl"
                    
                    # Check if the metadata file exists in the metadata list
            if metadata_file in mdata_list:
                        # Append the tuple (index_file, metadata_file, label) to the list
                metadata = read_mdata(metadata_file)
                st.session_state.index_metadata_label_list.append((idx, metadata, label))
        st.sidebar.success("Database Read!")    

    if user_data is not None:
        user_dataname = user_data.name
        user_idxname = f"index_{user_dataname[:-4]}.index"
        user_mdataname = f"metadata_{user_dataname[:-4]}.pkl"

        if os.path.isfile(user_idxname):
            st.sidebar.warning(f"File '{user_dataname}' already processed!")
        else:    
            with st.spinner(f"Processing '{user_dataname}', please wait..."):
                start_time = time.time()
                raw_extracted_text = extract_text_from_pdf(user_data)
                qa_pairs = extract_qa_pairs(raw_extracted_text)
                idx, metadata = store_in_faiss(qa_pairs, client)
                if not os.path.isfile(user_idxname):
                    store_idx(idx, user_idxname)
                if not os.path.isfile(user_mdataname):
                    store_mdata(metadata, user_mdataname)

                elapsed_time = time.time() - start_time
                st.sidebar.success(f"Processing for '{user_dataname}' completed in {str(round(elapsed_time,2))} seconds.")

    st.markdown(input_box_style, unsafe_allow_html=True)

    with st.form("input_form"):
        input_col, button_col = st.columns([5,1])
        with input_col:  
    
            user_input = get_user_message()
            st.session_state.user_input = user_input

        with button_col:
            st.markdown(
                "<div style='margin-top: 26px;'>",
                unsafe_allow_html=True
            )

            st.session_state.ai_response = ""
            submitted =  st.form_submit_button(use_container_width=True)

            if user_input and submitted:

                final_prompt = user_input
                st.session_state.messages.append(f"User: {user_input}")

                                        # Get OpenAI response
                related_chunks = search_across_multiple_indices(final_prompt, st.session_state.index_metadata_label_list, client, int(top_k))
                ai_response_list = []
                only_response_list = []

                for chunk in related_chunks:
                    context = chunk['context']
                    distance = chunk['distance']
                    label = chunk['label']
                    ai_response = get_openai_response(context, final_prompt, client)
                    ai_response_list.append({"aiResponse": ai_response, "context": context, "distance": distance, "label": label})
                    only_response_list.append(ai_response)

                for ai_response_dict in ai_response_list:
                    st.session_state.context_store[ai_response_dict["aiResponse"]] = ai_response_dict["context"]
                    st.session_state.distance_store[ai_response_dict["aiResponse"]] = ai_response_dict["distance"]
                    st.session_state.label_store[ai_response_dict["aiResponse"]] = ai_response_dict["label"]

                for response in only_response_list:
                    st.session_state.messages.append(f"AI: {response}")

                #st.session_state.ai_response = ai_response

                                        # Clear input and refresh the app
                st.session_state.user_input = ''

    display_message()
    st.write("\n" * 5)

if __name__ == "__main__":
    main()