import streamlit as st

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
@st.cache_resource()
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

generator = load_model()

# Define the LLM query function
def ask_local_llm(query, retrieved_chunks):
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    prompt = f"""
    You are a financial AI answering based on Cognizant's 2023 and 2024 report.
    Stick to the retrieved context. If unsure, say "I don't know."

    Context:
    {context}

    Question: {query}
    """
    response = generator(prompt, max_new_tokens=200)
    return response[0]["generated_text"]

# Placeholder retrieved chunks (to be replaced with actual retrieval logic)
retrieved_chunks = [{"text": "Cognizant's revenue grew by 5% in 2023."},
                    {"text": "The company's AI investments increased significantly in 2024."}]

# Set the page title
st.set_page_config(page_title="Financial RAG Chatbot Cognizant")

# Add a title to the app
st.title("Financial RAG Chatbot Cognizant")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create a text input for the user message
user_input = st.chat_input("Type your message...")

# If user sends a message
if user_input:
    # Add the user's message to the chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Hardcoded bot response
    bot_response = ask_local_llm(user_input, retrieved_chunks)

    # Add the bot's response to the chat history
    st.session_state.messages.append({"role": "bot", "content": bot_response})

    # Rerun the script to reflect the updated messages
    st.rerun()
