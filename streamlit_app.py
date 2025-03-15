import streamlit as st

# Set the page title
st.set_page_config(page_title="Messenger Chatbot")

# Add a title to the app
st.title("Messenger Chatbot")

# Create a function for the chatbot's response (you can replace this with an actual chatbot logic later)
def get_bot_response(user_message):
    return f"Bot: I received your message: {user_message}"

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display the chat history
for message in st.session_state['messages']:
    st.chat_message(message['role'], avatar_style="big-smile").markdown(message['content'])

# Get user input for the chatbot
user_input = st.text_input("Type your message:")

# If the user submits a message, store it and get a response
if user_input:
    st.session_state['messages'].append({"role": "user", "content": user_input})
    bot_response = get_bot_response(user_input)
    st.session_state['messages'].append({"role": "bot", "content": bot_response})

