import os
import openai
from openai import OpenAI
import streamlit as st
from constants import WRAPPER_PROMPT, MAIN_PAGE_STYLE, INPUT_BOX_STYLE
from chatbot_func import get_user_data, get_user_message, get_user_key, set_openai_client, display_message, get_embedding, search_in_faiss, get_openai_response, get_top_k
import pandas as pd
import time
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import sklearn
import faiss
import pickle

my_key = os.environ.get('MY_API_KEY','')
wrapper_prompt = WRAPPER_PROMPT
main_style = MAIN_PAGE_STYLE
input_box_style = INPUT_BOX_STYLE
index = faiss.read_index("faiss_index_2004_davidson.index")
with open('metadata_2004_davidson.pkl', 'rb') as f:
    metadata = pickle.load(f)

def main():
    st.markdown(main_style, unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    if 'ai_response' not in st.session_state:
        st.session_state.ai_response = ""

    if "context_store" not in st.session_state:
        st.session_state.context_store = {}

    if "distance_store" not in st.session_state:
        st.session_state.distance_store = {}

    user_key = get_user_key()
    client = set_openai_client(user_key)
    data = get_user_data()
    top_k = get_top_k()
    user_data = None
    if data is not None:
        user_data = pd.read_csv(data)
    
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
                related_chunks = search_in_faiss(final_prompt, index, metadata, client, int(top_k))

                ai_response_list = []
                only_response_list = []

                for chunk in related_chunks:
                    context = chunk[0]
                    distance = chunk[1]
                    ai_response = get_openai_response(context, final_prompt, client)
                    ai_response_list.append({"aiResponse": ai_response, "context": context, "distance": distance})
                    only_response_list.append(ai_response)

                for ai_response_dict in ai_response_list:
                    st.session_state.context_store[ai_response_dict["aiResponse"]] = ai_response_dict["context"]
                    st.session_state.distance_store[ai_response_dict["aiResponse"]] = ai_response_dict["distance"]

                for response in only_response_list:
                    st.session_state.messages.append(f"AI: {response}")

                #st.session_state.ai_response = ai_response

                                        # Clear input and refresh the app
                st.session_state.user_input = ''

    display_message()
    time.sleep(2)
    st.write("\n" * 5)

if __name__ == "__main__":
    main()