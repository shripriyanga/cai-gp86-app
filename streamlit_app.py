import streamlit as st
from random import uniform

def get_answer_and_confidence(query):
    """
    Simulate fetching an answer and confidence score for a given query.
    :param query: User input query.
    :return: A tuple of (answer, confidence_score).
    """
    # Simulate answer retrieval and confidence scoring
    answer = f"This is a simulated answer to: '{query}'"
    confidence_score = round(uniform(0.7, 1.0), 2)  # Simulated confidence score
    return answer, confidence_score

# Streamlit UI setup
st.set_page_config(page_title="Chatbot", page_icon="💬", layout="centered")

# Chat title and introduction
st.markdown("<h1 style='text-align: center; color: #333;'>💬 Chatbot Interface</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>How may I help you?</p>", unsafe_allow_html=True)

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input area
with st.form(key="chat_form"):
    user_query = st.text_input("Your message", placeholder="Type your query here...", label_visibility="collapsed")
    submit_button = st.form_submit_button(label="Send")

# Handle query submission
if submit_button and user_query.strip():
    # Fetch answer and confidence score
    answer, confidence = get_answer_and_confidence(user_query)

    # Save chat history
    st.session_state.chat_history.append({
        "query": user_query,
        "answer": answer,
        "confidence": confidence
    })

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    for chat in st.session_state.chat_history:
        st.markdown(f"<div style='background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>"
                    f"<b>You:</b> {chat['query']}<br>"
                    f"<b>Bot:</b> {chat['answer']}<br>"
                    f"<small style='color: #666;'>Confidence: {chat['confidence'] * 100:.0f}%</small>"
                    f"</div>", unsafe_allow_html=True)

# Exit option
if st.button("End Chat"):
    st.session_state.chat_history = []
    st.success("Chat session ended. Thank you for chatting!")
