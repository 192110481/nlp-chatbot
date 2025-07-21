import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer (using a smaller model for faster loading)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

tokenizer, model = load_model()

# Store past user inputs and responses
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_inputs" not in st.session_state:
    st.session_state.past_inputs = []

st.title("🧠 Simple AI Chatbot")
user_input = st.text_input("You:", key="input")

if user_input:
    # Encode the new user input, add the chat history tokens
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_input_ids

    # Generate a response
    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode and display
    response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    st.session_state.past_inputs.append((user_input, response))

# Display chat history
for user_text, bot_text in st.session_state.past_inputs:
    st.write(f"**You:** {user_text}")
    st.write(f"**Bot:** {bot_text}")
