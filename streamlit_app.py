import streamlit as st
import random

# Set the page title
st.set_page_config(page_title="Interactive Messenger Chatbot")

# Add a title to the app
st.title("Interactive Messenger Chatbot")

# Function to simulate getting a response from the chatbot
def get_bot_response(user_message):
    # Simulating an answer (you can replace this with a real model or API)
    answers = [
        "The answer to your query is 42.",
        "Let me look into that for you.",
        "Sorry, I don't have an answer to that right now.",
        "That's an interesting question, I'll get back to you with more info."
    ]
    answer = random.choice(answers)
    confidence_score = round(random.uniform(0.7, 1.0), 2)  # Random confidence score
    return answer, confidence_score

# Initialize session state for messages and bot responses
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display the chat history
for message in st.session_state['messages']:
    if message['role'] == 'user':
        st.chat_message(message['role'], avatar_style="big-smile").markdown(f"**{message['content']}**")
    else:
        st.chat_message(message['role'], avatar_style="big-smile").markdown(f"**Answer:** {message['content'][0]}")
        st.markdown(f"Confidence Score: **{message['content'][1]}**")

# Get user input for the chatbot
user_input = st.text_input("Type your message:")

# If the user submits a message, store it and get a response
if user_input:
    # Add the user's message to the chat history
    st.session_state['messages'].append({"role": "user", "content": user_input})
    
    # Get the bot response and confidence score
    bot_response, confidence = get_bot_response(user_input)
    
    # Add the bot's response to the chat history
    st.session_state['messages'].append({"role": "bot", "content": (bot_response, confidence)})

