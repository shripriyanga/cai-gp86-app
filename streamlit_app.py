import streamlit as st

# Streamlit Page Configuration
st.set_page_config(page_title="Chat Bot via Streamlit", page_icon=":speech_balloon:", layout="centered")

# Custom CSS for styling
def set_custom_css():
    st.markdown(
        """
        <style>
        .chat-container {
            max-width: 600px;
            margin: 0 auto;
        }
        .chat-header {
            text-align: center;
            font-size: 2rem;
            font-weight: bold;
            color: #6A1B9A;
        }
        .chat-bubble {
            border-radius: 20px;
            padding: 15px;
            margin: 10px 0;
            max-width: 80%;
        }
        .ai-bubble {
            background-color: #F1F1F1;
            color: #000;
            align-self: flex-start;
        }
        .user-bubble {
            background-color: #6A1B9A;
            color: #FFF;
            align-self: flex-end;
        }
        .thankyou-message {
            margin-top: 20px;
            font-style: italic;
            text-align: center;
            color: #555;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_custom_css()

# Chat Header
st.markdown('<div class="chat-header">Chat Bot via Streamlit</div>', unsafe_allow_html=True)

# Chat Container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Chat Messages
user_input_1 = "Hey"
ai_response_1 = "Hey there! How may I help you today?"
user_input_2 = "Suggest one good quote - just one"
ai_response_2 = (
    "\"The future belongs to those who believe in the beauty of their dreams.\" - Eleanor Roosevelt\n\n"
    "This quote is a reminder that anything is possible if you believe in yourself and your dreams. It is a powerful message that can inspire people to achieve great things."
)
user_input_3 = "Thank my readers"
ai_response_3 = (
    "Thank you for reading my work! I appreciate your support and feedback. I hope you enjoyed reading my work as much as I enjoyed writing it."
)

# Display Messages in Chat Layout
st.markdown(f'<div class="chat-bubble ai-bubble">{ai_response_1}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="chat-bubble user-bubble">{user_input_2}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="chat-bubble ai-bubble">{ai_response_2}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="chat-bubble user-bubble">{user_input_3}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="chat-bubble ai-bubble">{ai_response_3}</div>', unsafe_allow_html=True)

# Thank You Message
st.markdown(
    '<div class="thankyou-message">Thank you for interacting with this chatbot!</div>', unsafe_allow_html=True
)

# Close Chat Container
st.markdown('</div>', unsafe_allow_html=True)
