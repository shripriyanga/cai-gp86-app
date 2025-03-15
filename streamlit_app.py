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
st.title("Interactive Chatbot with Turn-Based Interaction")

# Entry and exit mechanism
if "chat_active" not in st.session_state:
    st.session_state.chat_active = False

if not st.session_state.chat_active:
    if st.button("Start Chat"):
        st.session_state.chat_active = True
        st.session_state.chat_history = []  # Initialize chat history
        st.session_state.current_turn = "User 1"
else:
    if st.button("End Chat"):
        st.session_state.chat_active = False
        st.success("Chat session ended. Thank you for chatting!")

# Chat logic
if st.session_state.chat_active:
    st.sidebar.header("Chat Options")
    st.sidebar.write("Type your question in the chat below and take turns.")

    # Display chat history
    st.subheader("Chat History")
    for i, chat in enumerate(st.session_state.chat_history):
        st.write(f"**{chat['user']}:** {chat['query']}")
        if chat['response']:
            st.write(f"**Bot:** {chat['response']}")
            st.progress(chat['confidence'])
            st.write(f"Confidence: {chat['confidence'] * 100:.0f}%")
        if i < len(st.session_state.chat_history) - 1:
            st.markdown("---")

    # Input field for user query
    user_query = st.text_input(f"{st.session_state.current_turn} (Your turn):", placeholder="Type your query here...")

    if st.button("Send"):
        if user_query.strip():
            if st.session_state.current_turn == "User 1":
                # Add User 1's query to the chat history
                st.session_state.chat_history.append({
                    "user": "User 1",
                    "query": user_query,
                    "response": None,
                    "confidence": None
                })
                st.session_state.current_turn = "User 2"
            else:
                # Fetch answer and confidence score for User 2's query
                answer, confidence = get_answer_and_confidence(user_query)
                st.session_state.chat_history.append({
                    "user": "User 2",
                    "query": user_query,
                    "response": answer,
                    "confidence": confidence
                })
                st.session_state.current_turn = "User 1"
        else:
            st.warning("Please enter a valid query before sending.")

    # Exit prompt
    if st.button("Exit Chat"):
        if st.radio("Do you want to exit?", ("Yes", "No")) == "Yes":
            st.session_state.chat_active = False
            st.success("Chat session ended. Have a great day!")

# Ensure responsiveness with a clean layout
st.markdown("---")
st.caption("Powered by Streamlit")
