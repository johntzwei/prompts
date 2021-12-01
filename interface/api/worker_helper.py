from typing import List
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer

def init(mc='t5-large'):
    global device, model, tokenizer, softmax, model_card

    model_card = mc

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.get_device_name(0))

    model = AutoModelWithLMHead.from_pretrained(model_card).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_card)
    model.eval()

    softmax = torch.nn.Softmax()

def inference(inputs: str, choices: List[str]):

    if model_card == 't5-large':
        return t5_inference(inputs, choices)


def t5_inference(inputs: str, choices: List[str]):
    # tokenize the choices
    # starting symbols should be added by tokenizer
    tokenized_choices = []
    for choice in choices:
        c = tokenizer(choice).input_ids
        tokenized_choices.append(c[:-1])
    max_choice_len = max(len(i) for i in tokenized_choices)

    inputs = tokenizer(inputs, return_tensors="pt", max_length=512).to(device)
    outputs = model.generate(**inputs, min_length=max_choice_len+1, max_length=max_choice_len+1, \
            return_dict_in_generate=True, output_scores=True)

    # get the scores for each of the words
    probs = []
    for choice in tokenized_choices:
        total_prob = 1
        for tok_logits, tok in zip(outputs.scores, choice):
            tok_dist = softmax(tok_logits)
            prob = tok_dist[0, tok]
            total_prob *= prob
        probs.append(total_prob.item())

    return probs
