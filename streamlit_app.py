import streamlit as st
from random import uniform

# st.title("🎈 My new app")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )

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
st.title("Interactive Query Interface")

st.sidebar.header("Query Options")
st.sidebar.write("Enter your query in the text box below and click 'Submit' to get an answer.")

# Input field for user query
user_query = st.text_input("Enter your query:", placeholder="Type your question here...")

if st.button("Submit"):
    if user_query.strip():
        # Fetch answer and confidence score
        answer, confidence = get_answer_and_confidence(user_query)
        
        # Display results
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Confidence Score")
        st.progress(confidence)
        st.write(f"Confidence: {confidence * 100:.0f}%")
    else:
        st.warning("Please enter a valid query before submitting.")

# Ensure responsiveness with a clean layout
st.markdown("---")
st.caption("Powered by Streamlit")
