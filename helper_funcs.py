import random
import numpy as np
import pickle
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    # Loading text embs
    #print('\nLoading embeddings')
    
    words = load_all_words(train_file_path)
    word_idx = {word: idx + 1 for idx, word in enumerate(words)}
    word_idx_rev = {idx + 1: word for idx, word in enumerate(words)}

    w2v = load_glove(embedding_path)
    embedding = init_embs(words, w2v, embedding_dim)

    embedding_pos = init_pos_embs(embedding_dim_pos)
    #print('Embeddings loaded.')

    return embedding, embedding_pos, word_idx, word_idx_rev

def normalize_embeddings(embeddings):
    # Normalize embeddings
    data = embeddings[1:, :]  # Exclude the first row
    min_vals = torch.min(data, dim=0, keepdim=True).values
    max_vals = torch.max(data, dim=0, keepdim=True).values
    data = (data - min_vals) / (max_vals - min_vals + 1e-8)
    embeddings[1:, :] = data
    return embeddings