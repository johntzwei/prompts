import torch
import ipdb
import json
import csv
import sys
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelWithLMHead, AutoTokenizer

# read in csv file
fn = sys.argv[1]
print(fn)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = AutoModelWithLMHead.from_pretrained("t5-large").to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("t5-large")

softmax = torch.nn.Softmax()

df = pd.read_csv(fn)
df.columns = ['index', 'input', 'output', 'fixed_choices', 'label']

# T5 uses a max_length of 512 so we cut the article to 512 tokens.
basename = fn.split('/')[-1][:-4]
data = open('outputs/%s.jsonl' % basename, 'wt')
for i, row in tqdm(df.iterrows()):

    # get all answer choices
    choices = json.loads(row['fixed_choices'])
    tokenized_choices = []
    for choice in choices:
        c = tokenizer(choice).input_ids
        tokenized_choices.append(c[:-1])
    max_choice_len = max(len(i) for i in tokenized_choices)

    inputs = tokenizer(row['input'], return_tensors="pt", max_length=512).to(device)
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

    data.write(json.dumps(probs) + '\n')
