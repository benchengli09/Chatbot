WRAPPER_PROMPT = """
Respond to the user prompt describing how they can achieve their task in Python, 
include Python code in one single code that will achieve the task. 
This is the user prompt input: 
"""

MAIN_PAGE_STYLE = """
        <style>
        .stApp {
            flex-direction: column;
            overflow-y: auto;
        }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            margin-bottom: 100px; /* Space for input box */
            margin-top: -100px;
        }
        .user-message {
            background-color: #F0F2F6;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            width: fit-content;
            animation: fadeIn 2s ease-in;
        }
        .ai-message {
            background-color: #F0F2F6;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            width: fit-content;
            animation: fadeIn 2s ease-in
        }
        # @keyframes fadeIn {
        #     0% {opacity: 0;}
        #     100% {opacity: 1;}
        # }

        </style>
        """
INPUT_BOX_STYLE = """
        <style>
        div[data-testid="stForm"]{
            position:fixed;
            right: 23%;
            left: 41%;
            bottom: 8%;
            background-color: #F0F2F6;
            padding: 10px;
            z-index: 10;
        }
        </style>
        """


# {response: [{aiResponse: '', similarity: '', reference: ''}, {}, {}]}