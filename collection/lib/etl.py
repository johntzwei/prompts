from datasets import load_dataset
import json

def etl(name):
    if name == 'sst2':
        raw = load_dataset("glue", "sst2")['train']
        examples = raw['sentence']
        labels = raw['label']

    data = {
        'labels': labels,
        'examples': examples
        }

    return data
