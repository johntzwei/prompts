import json
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import numpy as np
from lib.etl import etl

def get_accuracy(preds, labels, idxs):
    preds = [preds[i] for i in idxs]
    labels = [labels[i] for i in idxs]
    rounded = [round(p) for p in preds]
    agreement = [ 1 - (pair[0] ^ pair[1]) for pair in list(zip(labels, rounded))]
    return sum(agreement) / len(agreement)

def extract_features(preds_pair):
    pos, neg = preds_pair
    return (pos + neg, pos / (pos + neg))

def analyze(state_dir, adversary_file, prompts_file, split_file, cache_file, dataset):
    print("Analyzing output...")
    print()

    with open(state_dir + prompts_file, 'r') as infile:
        prompts = json.load(infile)[dataset]

    with open(state_dir + split_file, 'r') as infile:
        split = json.load(infile)[dataset]

    with open(state_dir + cache_file, 'r') as infile:
        cache = json.load(infile)[dataset]

    data = etl(dataset)
    labels = data['labels']

    calibrated = {}
    for prompt in prompts:
        prompt_output = cache[prompt]
        preds = [ (pair[0] / (pair[0] + pair[1])) for pair in prompt_output ]
        print(f"Prompt: {prompt} has uncalibrated accuracy: {get_accuracy(preds, labels, split['test'])}")
        calib = {
            'labels': [ labels[i] for i in split['calibration'] ],
            'data': [ extract_features(prompt_output[i]) for i in split['calibration'] ]
            }
        calibrator = LogisticRegression(random_state=0).fit(calib['data'], calib['labels'])
        calib_preds = list(calibrator.predict_proba([ extract_features(p) for p in prompt_output ])[:, 1])
        print(f"And has calibrated accuracy: {get_accuracy(calib_preds, labels, split['test'])}")
        calibrated[prompt] = calib_preds

    
    features = list(zip(*[ preds for _, preds in calibrated.items() ]))
    agg_training = {
        'labels': [ labels[i] for i in split['aggregation_training'] ],
        'data': [ features[i] for i in split['aggregation_training'] ]
    }
    aggregator = LogisticRegression().fit(agg_training['data'], agg_training['labels'])
    aggregated_preds = list(aggregator.predict_proba(features)[:, 1])

    print(f"Aggregated accuracy is {get_accuracy(aggregated_preds, labels, split['test'])}")
    
    point_accuracy = 1 - np.abs(np.subtract(aggregated_preds, labels))    
    sorted = np.argsort(point_accuracy)                    
    adversary_set = set(split['adversarial'])
    sorted_adversaries = list(filter(lambda x : x in adversary_set, sorted))
    serializable = [ int(adv) for adv in sorted_adversaries ]

    with open(state_dir + adversary_file, 'w') as outfile:
        json.dump({ dataset: serializable }, outfile)
