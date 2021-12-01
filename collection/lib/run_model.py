import json
from os import truncate
import torch
from tqdm import tqdm
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer
from os.path import exists
from lib.etl import etl

from transformers.tokenization_utils_base import TruncationStrategy

model_name = 'roberta-base'
batch_size = 32

def parse_prompt(tokenizer, raw):
    char_arr = list(raw)
    langle = char_arr.index("<")
    slash = char_arr.index("/")
    rangle = char_arr.index(">")

    masked = ''.join(char_arr[0 : langle]) + tokenizer.mask_token + ''.join(char_arr[rangle + 1 : len(char_arr) - 1])

    targets = [char_arr[langle + 1 : slash], char_arr[slash + 1 : rangle]]

    target_tokens = [''.join(["Ġ"] + t) for t in targets]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    return masked, target_ids

def chunks(arr, chunk_size):
    for i in range(0, len(arr), chunk_size):
        yield arr[i:i + chunk_size]

def evaluate(model, tokenizer, device, examples, raw_prompt):

    print(f"Running on prompt: \"{raw_prompt}\"")
    n = len(examples)
    softmax = torch.nn.Softmax(dim=0)

    parsed, target_ids = parse_prompt(tokenizer, raw_prompt)

    inputs = list(chunks([(e + tokenizer.sep_token + parsed) for e in examples], batch_size))
    output_probs = []

    for batch in tqdm(inputs):
        tokens = tokenizer(batch, return_tensors='pt', padding=True).to(device)
        mask_positions = [ x[1] for x in
                            torch.nonzero(tokens.input_ids == tokenizer.mask_token_id).cpu().numpy() ]

        output = model(**tokens)['logits']

        for i in range(len(mask_positions)):
            all_probs = softmax(output[i][mask_positions[i]]).cpu().detach().numpy()
            output_probs.append([ all_probs[t].item() for t in target_ids])

    return output_probs

def run_model(state_dir, dataset, prompts_file, output_cache_file):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"Running prompts on model... (using {device})")
    print()
    examples = etl(dataset)['examples']

    with open(state_dir + prompts_file, 'r') as infile:
        raw_prompts = json.load(infile)[dataset]

    if exists(state_dir + output_cache_file):
        with open(state_dir + output_cache_file, 'r') as infile:
            cache = json.load(infile)
    else:
        cache = {}

    if not dataset in cache:
        cache[dataset] = {}

    model = RobertaForMaskedLM.from_pretrained(model_name)
    model.to(device)
    tokenizer = RobertaTokenizer.from_pretrained(model_name, truncation_side='left')

    for pr in raw_prompts:
        if not pr in cache[dataset]:
            cache[dataset][pr] = evaluate(model, tokenizer, device, examples, pr)

    with open(state_dir + output_cache_file, 'w') as outfile:
        cache = json.dump(cache, outfile)
