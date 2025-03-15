import streamlit as st

# Set the page title
st.set_page_config(page_title="Interactive Messenger Chatbot")

# Add a title to the app
st.title("Interactive Messenger Chatbot")

# Initialize session state for messages and input message
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'input_message' not in st.session_state:
    st.session_state['input_message'] = ""

# Display the chat history
for message in st.session_state['messages']:
    if message['role'] == 'user':
        st.chat_message(message['role']).markdown(f"**{message['content']}**")
    else:
        st.chat_message(message['role']).markdown(f"**Bot's Response:** {message['content']}")

# Create a text input for the user message
user_input = st.text_area("Type your message:", height=100, value=st.session_state['input_message'])

# Add a submit button
submit_button = st.button("Submit")

# If the user submits a message, store it and get a hardcoded response
if submit_button and user_input:
    # Add the user's message to the chat history
    st.session_state['messages'].append({"role": "user", "content": user_input})
    
    # Hardcoded bot response
    bot_response = "This is a hardcoded response from the bot."
    
    # Add the bot's response to the chat history
    st.session_state['messages'].append({"role": "bot", "content": bot_response})
    
    # Clear the input after submission
    st.session_state['input_message'] = ""  # Reset input_message session state

    # Prevent the previously entered message from persisting
    st.text_area("Type your message:", height=100, value="")  # Clear the input field immediately

