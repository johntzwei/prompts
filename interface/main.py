import streamlit as st
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelWithLMHead


@st.cache
def load_model(model_name):
    model = AutoModelWithLMHead.from_pretrained(model_name)
    return model


@st.cache
def run(name):
    """ infer and display result 
    """
    print("name: ", name)
    # forward pass
    prompt = "My name is " + name
    encoded_prompt = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt")
    output_sequences = model.generate(input_ids=encoded_prompt)
    # decode the output sequences
    generated_text = []
    for output_sequence in output_sequences:
        output_sequence = output_sequence.tolist()
        text = tokenizer.decode(
            output_sequence, clean_up_tokenization_spaces=True)
        generated_text.append(text)
    print("####### generated_text: ", generated_text)
    # display the generated text
    st.write(generated_text)


def run_all(input_list):
    for name in input_list:
        if st.session_state[name]:
            run(st.session_state[name])


# load models
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = load_model(model_name)


st.text_input("Your name", key="name")
st.text_input("Your name", key="name2")
st.text_input("Your name", key="name3")
st.text_input("Your name", key="name4")

run_all(["name", "name2", "name3", "name4"])


# if st.session_state.name:
#     run(st.session_state.name)
