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
def init_embs(words, w2v, embedding_dim):
    # Initialize embedding matrix
    embedding_matrix = [np.zeros(embedding_dim)]  # Add zero vector for padding
    for word in words:
        if word in w2v:
            embedding_matrix.append(w2v[word])
        else:
            embedding_matrix.append(np.random.uniform(-0.1, 0.1, embedding_dim))
    #print(f'{len(embedding_matrix)} words have embeddings')
    return np.array(embedding_matrix)

def init_pos_embs(embedding_dim_pos, max_position=200):
    # Pos embeddings random init
    embedding_pos = [np.zeros(embedding_dim_pos)]
    embedding_pos.extend(np.random.normal(0.0, 0.1, (max_position, embedding_dim_pos)))
    return np.array(embedding_pos)