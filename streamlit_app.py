import streamlit as st
from streamlit_chat import message

# Page configuration
st.set_page_config(page_title="Chat Bot via Streamlit", layout="centered")

# Title
st.markdown("# Chat Bot via Streamlit")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to simulate bot response
def get_bot_response(user_input):
    if user_input.lower() == "suggest one good quote - just one":
        return "\"The future belongs to those who believe in the beauty of their dreams.\" - Eleanor Roosevelt"
    elif user_input.lower() == "thank my readers":
        return "Thank you for reading my work! I appreciate your support and feedback. I hope you enjoyed reading my work as much as I enjoyed writing it."
    else:
        return "I'm sorry, I don't have an answer for that. Could you ask something else?"

# Chat interface
with st.container():
    # Display existing conversation
    for i, message_data in enumerate(st.session_state.messages):
        message(message_data["content"], is_user=message_data["is_user"], key=f"msg_{i}")

    # Input box for user message
    user_input = st.text_input("Type your message:", key="user_input")

    if user_input:
        # Append user message to history
        st.session_state.messages.append({"content": user_input, "is_user": True})

        # Get bot response and append to history
        bot_response = get_bot_response(user_input)
        st.session_state.messages.append({"content": bot_response, "is_user": False})

        # Clear the input box
        st.experimental_rerun()
