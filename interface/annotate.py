import streamlit as st
import requests

server_url = "http://127.0.0.1:8000/evaluate-prompt/"

prompt_num = 1

st.text_input(f"prompt{prompt_num}", key=f"prompt{prompt_num}")

# TODO: adding prompt doesn't work -- might be limitation in Streamlit itself
if st.button('Add a Prompt'):
    prompt_num += 1
    st.text_input(f"prompt{prompt_num}", key=f"prompt{prompt_num}")


def evaluate_prompt(server_url: str, prompt: str):
    data = {"prompt": prompt}
    r = requests.post(server_url, json=data)
    return r


if st.session_state[f'prompt{prompt_num}']:
    res = evaluate_prompt(server_url, st.session_state[f'prompt{prompt_num}'])
    st.write(res.content)
