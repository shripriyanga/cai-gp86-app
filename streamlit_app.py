import streamlit as st

# Set the page title
st.set_page_config(page_title="Interactive Messenger Chatbot")

# Add a title to the app
st.title("Interactive Messenger Chatbot")

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display the chat history
for message in st.session_state['messages']:
    if message['role'] == 'user':
        st.chat_message(message['role'], avatar_style="big-smile").markdown(f"**{message['content']}**")
    else:
        st.chat_message(message['role'], avatar_style="big-smile").markdown(f"**Bot's Response:** {message['content']}")

# Get user input for the chatbot
user_input = st.text_input("Type your message:")

# If the user submits a message, store it and get a hardcoded response
if user_input:
    # Add the user's message to the chat history
    st.session_state['messages'].append({"role": "user", "content": user_input})
    
    # Hardcoded bot response
    bot_response = "This is a hardcoded response from the bot."
    
    # Add the bot's response to the chat history
    st.session_state['messages'].append({"role": "bot", "content": bot_response})
