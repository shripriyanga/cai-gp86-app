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
st.title("Interactive Chatbot Interface")

# Entry and exit mechanism
if "chat_active" not in st.session_state:
    st.session_state.chat_active = False

if not st.session_state.chat_active:
    if st.button("Start Chat"):
        st.session_state.chat_active = True
        st.session_state.chat_history = []  # Initialize chat history
else:
    if st.button("End Chat"):
        st.session_state.chat_active = False
        st.success("Chat session ended. Thank you for chatting!")

# Chat logic
if st.session_state.chat_active:
    st.sidebar.header("Chat Options")
    st.sidebar.write("Type your question in the chat below and click 'Send' to get a response.")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input field for user query
    user_query = st.text_input("Your turn:", placeholder="Type your question here...")

    if st.button("Send"):
        if user_query.strip():
            # Fetch answer and confidence score
            answer, confidence = get_answer_and_confidence(user_query)

            # Save chat history
            st.session_state.chat_history.append({
                "query": user_query,
                "answer": answer,
                "confidence": confidence
            })
        else:
            st.warning("Please enter a valid query before sending.")

    # Display chat history
    st.subheader("Chat History")
    for i, chat in enumerate(st.session_state.chat_history):
        st.write(f"**You:** {chat['query']}")
        st.write(f"**Bot:** {chat['answer']}")
        st.progress(chat['confidence'])
        st.write(f"Confidence: {chat['confidence'] * 100:.0f}%")
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("---")

# Ensure responsiveness with a clean layout
st.markdown("---")
st.caption("Powered by Streamlit")
