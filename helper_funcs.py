import random
import numpy as np
import pickle
import json

def load_pickle_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def load_json_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def load_all_words(filename):
    # Extract words from train file
    words = set()
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().split(' | ')
            emotion, clause = line[2], line[-1]
            words.update([emotion] + clause.split())
    return words

def load_glove(embedding_path):
    w2v = {}
    with open(embedding_path, 'r') as file:
        file.readline()  # Skip header
        for line in file:
            parts = line.strip().split(' ')
            word, embedding = parts[0], list(map(float, parts[1:]))
            w2v[word] = embedding
    #print(f'{len(w2v)} words loaded from glove')
    return w2v
