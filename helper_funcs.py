import random
import numpy as np
import pickle
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

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

def load_mapping(mapping_file):
    # Load mapping
    mapping = np.load(mapping_file, allow_pickle=True).item()
    return mapping
def load_and_normalize_embeddings(emb_file, dtype=torch.float32):
    # Normalize embeddings
    embeddings = torch.tensor(np.load(emb_file, allow_pickle=True), dtype=dtype)
    embeddings = normalize_embeddings(embeddings)
    return embeddings

def load_video_audio_embeddings(video_mapping_file, video_emb_file, audio_emb_file):
    # Load video and audio embeddings
    #print("\nLoading embeddings...")

    #Load video ID mapping
    video_id_mapping = load_mapping(video_mapping_file)

    #Load and normalize video and audio embeddings
    video_embeddings = load_and_normalize_embeddings(video_emb_file)
    audio_embeddings = load_and_normalize_embeddings(audio_emb_file)

    # print(f"Loaded video embeddings: {video_emb_file}, shape: {video_embeddings.shape}")
    # print(f"Loaded audio embeddings: {audio_emb_file}, shape: {audio_embeddings.shape}\n")

    return video_id_mapping, video_embeddings, audio_embeddings
def adjust_sequence_length(sequences, max_utt, segment_indices, max_sen_len):
    # Adjust sequence length
    # Conditions: max 35 words per utterance, max 35 utterances per dialogue
    if segment_indices[max_sen_len] > max_utt:
        segment_indices = np.array(segment_indices)
        clause_max_utt = max_utt // max_sen_len
        new_segment_indices = np.zeros_like(segment_indices)
        adjusted_sequences = np.zeros_like(sequences)
        mask = np.zeros_like(sequences, dtype=int)
        
        pos = 0
        for i in range(max_sen_len):
            start, end = segment_indices[i], segment_indices[i + 1]
            segment_length = min(end - start, clause_max_utt)

            adjusted_sequences[pos:pos + segment_length] = sequences[start:start + segment_length]
            mask[pos:pos + segment_length] = 1
            new_segment_indices[i + 1] = new_segment_indices[i] + segment_length
            pos += segment_length

        # Truncate sequences and update segment indices
        sequences = adjusted_sequences[:max_utt]
        mask = mask[:max_utt]
        segment_indices = new_segment_indices.tolist()
    else:
        mask = np.zeros_like(sequences, dtype=int)
        mask[:segment_indices[max_sen_len]] = 1

    return sequences[:max_utt], mask, segment_indices
def load_data(input_file, word_idx, video_idx, spe_idx, max_doc_len, max_sen_len):
    """
    Load and process data for BiLSTM with speaker information.

    Args:
        input_file (str): Path to the input file.
        word_idx (dict): Mapping of words to indices.
        video_idx (dict): Mapping of video IDs to indices.
        spe_idx (dict): Mapping of speakers to indices.
        max_doc_len (int): Maximum number of utterances in a document.
        max_sen_len (int): Maximum number of tokens in a sentence.

    Returns:
        tuple: Processed data arrays and metadata.
    """
   # print(f'\nLoading data from: {input_file}\n')

    doc_id, y_emotion, y_cause, x, x_v, sen_len, doc_len, speaker, y_pairs, num_token = [[] for _ in range(10)]
    num_emo, num_pairs = 0, 0

    with open(input_file, 'r', encoding='utf-8') as file:
        while True:
            line = file.readline()
            if not line:
                break

            # Process document-level metadata
            line = line.strip().split()
            d_id, d_len = line[0], int(line[1])
            doc_id.append(d_id)
            doc_len.append(d_len)

            # Process emotion-cause pairs
            pairs = eval('[' + file.readline().strip() + ']')
            if pairs:
                pairs = sorted(list(set(pairs)))
                y_pairs.append(pairs)
                num_pairs += len(pairs)

            # Initialize arrays for the document
            y_emotion_tmp = np.zeros((max_doc_len, 2))  # Binary emotion labels
            y_cause_tmp = np.zeros((max_doc_len, 2))    # Binary cause labels
            x_tmp = np.zeros((max_doc_len, max_sen_len), dtype=np.int32)
            x_v_tmp = np.zeros(max_doc_len, dtype=np.int32)
            sen_len_tmp = np.zeros(max_doc_len, dtype=np.int32)
            spe_tmp = np.zeros(max_doc_len, dtype=np.int32)
    # Process each sentence in the document
            for i in range(d_len):
                line = file.readline().strip().split(' | ')
                x_v_tmp[i] = video_idx.get(f'dia{d_id}utt{i + 1}', 0)

                # Process speaker
                speaker_name = line[1]
                if speaker_name in spe_idx:
                    spe_tmp[i] = spe_idx[speaker_name]
                #else:
                   #print(f'Warning: Speaker "{speaker_name}" not found in index.')

                # Process emotion label
                emotion = line[2]
                y_emotion_tmp[i] = [1, 0] if emotion == 'neutral' else [0, 1]
                num_emo += (emotion != 'neutral')

                # Process cause label
                cause = [p[1] for p in pairs if p[0] == i + 1]
                y_cause_tmp[i][int(len(cause) > 0)] = 1

                # Process words in the sentence
                words = line[3].replace('|', '').split()
                sen_len_tmp[i] = min(len(words), max_sen_len)
                for j, word in enumerate(words[:max_sen_len]):
                    x_tmp[i][j] = word_idx.get(word, 0)

            # Append document data
            y_emotion.append(y_emotion_tmp)
            y_cause.append(y_cause_tmp)
            x.append(x_tmp)
            x_v.append(x_v_tmp)
            sen_len.append(sen_len_tmp)
            speaker.append(spe_tmp)
# Convert lists to numpy arrays
    x, x_v, sen_len, doc_len, speaker, y_emotion, y_cause = map(
        np.array, [x, x_v, sen_len, doc_len, speaker, y_emotion, y_cause]
    )

    return x, x_v, sen_len, doc_len, speaker, y_emotion, y_cause, doc_id, y_pairs