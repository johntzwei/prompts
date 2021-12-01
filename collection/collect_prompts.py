import json
from math import fabs
import numpy as np
import random
from lib.run_model import run_model
from lib.analyze import analyze
from lib.etl import etl

from pprint import pprint
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizer
from os.path import exists
import os

state_dir = "./state/"
adversary_file = "adversaries.json"
dataset = "sst2"
prompts_file = "prompts.json"
output_cache_file = "cache.json"
split_file = "split.json"
fig_folder = "./figs/"

split_proportions = {
  'calibration': 0.1,
  'aggregation_training': 0.1,
  'adversarial': 0.05
  }

def initialize():
  print("Initializing...\n")
  if not exists(state_dir + adversary_file):
    reset()

def reset():
  print("Resetting computation...\n")
  print()
  data = etl(dataset)

  n = len(data['examples'])

  adversary_idxs = { dataset: sorted(range(n), key=lambda x: random.random()) }
  with open(state_dir + adversary_file, 'w') as outfile:
    json.dump(adversary_idxs, outfile)

  with open(state_dir + prompts_file, 'w') as outfile:
    json.dump({}, outfile)

  all_idxs = list(range(n))
  split_idxs = {dataset: {}}

  for set_name, p in split_proportions.items():
    idxs = []
    for i in range(int(p * n)):
      idxs.append(all_idxs.pop(random.randrange(len(all_idxs))))
    split_idxs[dataset][set_name] = idxs
  split_idxs[dataset]['test'] = all_idxs

  with open(state_dir + split_file, 'w') as outfile:
    json.dump(split_idxs, outfile)


def query_input(prompt):
  inp = input(f"{prompt}\n> ")
  print()
  return inp

def well_formed(prompt, tokenizer):
  char_arr = list(prompt)
  if not ("/" in char_arr and "<" in char_arr and ">" in char_arr):
    print("Prompt must contain a mask token of the form: <target1/target2>\n")
    return False

  langle = char_arr.index("<")
  rangle = char_arr.index(">")
  slash = char_arr.index("/")

  targets = [char_arr[langle + 1 : slash], char_arr[slash + 1 : rangle]]
  target_tokens = [''.join(["Ġ"] + t) for t in targets]
  target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

  pprint(target_ids)
  for i in range(2):
    if target_ids[i] == 3:
      print(f"Target \"{target_tokens[i][1:]}\" is not a valid token.\n")
      return False

  return True

def collect(n, tokenizer):

  data = etl(dataset)

  with open(state_dir + adversary_file, 'r') as infile:
    adversary_idxs = json.load(infile)

  if exists(state_dir + prompts_file):
    with open(state_dir + prompts_file, 'r') as infile:
      prompts = json.load(infile)
      if not dataset in prompts:
        prompts[dataset] = []
  else:
    prompts = { dataset: []}

  print("********************************************************************************")
  print("For each example and label, provide a suitable prompt. Example format:\n\n\
    \"Overall, the movie was <good/bad>.\"\n\nThe positive target should come first.")
  print("********************************************************************************\n")
  
  for i in range(n):
    print(f"Example #{i+1}:\n\n\
    \"{data['examples'][adversary_idxs[dataset][i]]}\"\n")
    print(f"Label: {1}\n")

    while True:
      prompt = query_input(f"Please provide a suitable prompt for example #{i + 1}:")
      if well_formed(prompt, tokenizer):
        break
    prompts[dataset].append(prompt)
  
  with open(state_dir + prompts_file, 'w') as outfile:
    json.dump(prompts, outfile)

def interpret(tokenizer):
  err_msg = "Could not recognize command. Please only input a positive integer, or \"RESET\"."
  inp = query_input("Type the number of prompts you wish to supply, or \"RESET\" to reset the prompt set.")
  if inp == "RESET":
    reset()
    return False
  else:
    try:
      n = int(inp)
      if n >= 0:
        collect(n, tokenizer)
        return True
      else:
        print(err_msg)
        return False
    except ValueError:
      print(err_msg)
      return False

if __name__ == '__main__':
  initialize()
  random.seed(5)
  tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
  if interpret(tokenizer):
    run_model(state_dir, dataset, prompts_file, output_cache_file)
    analyze(state_dir, adversary_file, prompts_file, split_file, output_cache_file, dataset)
