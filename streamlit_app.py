import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb

import streamlit as st
from RAG_implementation import prep,process_and_store,retrieve_similar_chunks,ask_local_llm,ask_local_llm_adv,retrieve_similar_chunks_Advanced_Rag,GuardRail

# Initialize only once
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    # Process and store the financial reports
    prep()
    process_and_store()


# Set the page title
st.set_page_config(page_title="Financial RAG Chatbot Cognizant")

# Add a title to the app
st.title("Financial RAG Chatbot Cognizant")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display confidence score for bot responses
        if message["role"] == "bot" and "confidence" in message:
            with st.expander("Confidence Score"):
                st.markdown(f"**Confidence:** {message['confidence']:.2f}")


# Create a text input for the user message
user_input = st.chat_input("Type your message...")


#retrieved_chunks = [{"text": "Cognizant's revenue grew by 5% in 2023."},
#                    {"text": "The company's AI investments increased significantly in 2024."}]



# If user sends a message
if user_input:
    # Add the user's message to the chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    retrieved_chunks = GuardRail(user_input)

    if "Your query seems to contain the term" in retrieved_chunks:
        bot_response = retrieved_chunks
        confidence_score = 0
    #retrieved_chunks = retrieve_similar_chunks_Advanced_Rag(user_input)
    else:
        bot_response,confidence_score = ask_local_llm_adv(user_input, retrieved_chunks)

    # if "I'm currently specialized in financial reports and related topics. Could you please ask a financial-related question?" in bot_response:
    #     bot_response = "I'm currently specialized in financial reports and related topics. Could you please ask a financial-related question?"

    # Add the bot's response to the chat history
    st.session_state.messages.append({"role": "bot", "content": bot_response, "confidence": confidence_score})

    # Rerun the script to reflect the updated messages
    st.rerun()
