import json
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad
import torch
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
nltk.download('punkt')

class WAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(WAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, lstm_out):
        # attention scores
        attn_scores = self.attention(lstm_out).squeeze(-1)  # Shape: (batch_size, seq_len)
        attn_weights = F.softmax(attn_scores, dim=1)  # Shape: (batch_size, seq_len)

        #attention weights to LSTM output
        weighted_output = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)  # Shape: (batch_size, hidden_dim*2)
        return weighted_output, attn_weights

class WordLevelBiLSTM(nn.Module):
    def __init__(self, glove_embedding, embedding_dim, hidden_dim):
        super(WordLevelBiLSTM, self).__init__()
        glove_embedding = torch.Tensor(glove_embedding)
        self.embedding = nn.Embedding.from_pretrained(glove_embedding, freeze=True)
        #print(self.embedding, glove_embedding)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Hidden dimension divided by 2 since it's bidirectional
            batch_first=True,
            bidirectional=True
        )
        self.attention = WAttention(hidden_dim, hidden_dim)  # input_dim = hidden_dim since it's bidirectional
    
    def forward(self, x):
        #x = self.embedding(x)  # Shape: (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, seq_len, hidden_dim*2)
        weighted_output, attn_weights = self.attention(lstm_out)  # weighted_output: (batch_size, hidden_dim*2)
        return weighted_output, attn_weights

def load_glove_embeddings(embedding_path):
    print("Loading GloVe embeddings...")
    glove = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec
    return glove

def generate_embeddings_and_pos(glove, sentence, embedding_dim, pos_id_len=50):
    print("Generating embeddings and POS tags...")
    words = word_tokenize(sentence)
    embeddings = []
    pos_ids = []
    for i, word in enumerate(words):
        if word in glove:
            embeddings.append(glove[word])
        else:
            embeddings.append(np.random.rand(embedding_dim) / 5. - 0.1)
        pos_ids.append(i + 1)
    
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    pos_ids = torch.tensor(pos_ids, dtype=torch.float32)

    # if len(embeddings) < pos_id_len:
    #     padded_pos_ids = pad(pos_ids, (0, pos_id_len - len(pos_ids)), value=0)
    # else:
    #     padded_pos_ids = pos_ids[:pos_id_len]
    
    return embeddings, pos_ids






if __name__ == '__main__':
    # Load GloVe embeddings
    glove = load_glove_embeddings('../../data/glove.6B.300d.txt')
    # Load the file
    with open('../../data/train.json', 'r') as file:
        data = json.load(file)

    conversations = []
    for conv in data:
        conversation_text = " ".join([utterance['text'] for utterance in conv['conversation']])
        conversations.append(conversation_text)

    for i, conv in enumerate(conversations[:5]):
        print(f"Conversation {i + 1}:")
        print(conv,end="")
        print("\n" + "-"*50 + "\n")  

    # Tokenize the text
    tokenized_conversations = [word_tokenize(conv) for conv in conversations]
    print(tokenized_conversations)

    ex = tokenized_conversations[0]
    print(ex)
    embeddings, pos_ids = generate_embeddings_and_pos(glove, ex, 300)
    print(embeddings, pos_ids)

    # Initialize the BiLSTM model
    model = WordLevelBiLSTM(glove, 300, 128)
    weighted_output, attn_weights = model(embeddings.unsqueeze(0))

    # print(weighted_output.shape)
    # print(attn_weights.shape)
    # print(attn_weights)

    print("Weighted output shape:", weighted_output.shape)
    print("Attention weights shape:", attn_weights.shape)