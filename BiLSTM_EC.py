import argparse
import numpy as np
import random
from helper import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description='BiLSTM Training Configuration')
parser.add_argument('--features_dir', type=str, default='features/', help='Features directory')
parser.add_argument('--video_emb_file', type=str, default='video_emb.npy', help='Video embedding file')
parser.add_argument('--audio_emb_file', type=str, default='audio_emb.npy', help='Audio embedding file')
parser.add_argument('--word_emb_file', type=str, default='word_emb.npy', help='Word embedding file')
parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding dimension')
parser.add_argument('--pos_emb_dim', type=int, default=50, help='Position embedding dimension')
parser.add_argument('--max_words', type=int, default=35, help='Maximum words per sentence')
parser.add_argument('--max_utt', type=int, default=35, help='Maximum utterances per document')
parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden layer dimension')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--lrate', type=float, default=0.005, help='Learning rate')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--l2_reg', type=float, default=1e-5, help='L2 regularization lambda')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
args = parser.parse_args()

class Args:
    def __init__(self):
        self.features_dir = 'features/'
        self.video_emb_file = 'video_emb.npy'
        self.audio_emb_file = 'audio_emb.npy'
        self.word_emb_file = 'word_emb.npy'
        self.video_idx_file = './data/video_id_mapping.npy'
        self.word_emb_dim = 300
        self.pos_emb_dim = 50
        self.max_words = 35
        self.max_utt = 35
        self.hidden_dim = 100
        self.batch_size = 32
        self.lrate = 0.005
        self.dropout = 0.5
        self.l2_reg = 1e-5
        self.epochs = 20
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 2  # num_pred = (E,C)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

class BiLSTMModule(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, max_sen_len, max_doc_len, dropout):
        super(BiLSTMModule, self).__init__()
        self.max_sen_len = max_sen_len
        self.max_doc_len = max_doc_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.word_bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.sentence_bilstm = nn.LSTM(
            input_size=2 * hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.attention_layer = nn.Linear(2 * hidden_dim, 1)
        self.fc_emotion = nn.Linear(2 * hidden_dim, output_dim)
        self.fc_cause = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def attention(self, lstm_output, lengths):
        scores = self.attention_layer(lstm_output).squeeze(-1)
        mask = torch.arange(scores.size(1), device=scores.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=1)
        weighted_output = (lstm_output * attention_weights.unsqueeze(-1)).sum(dim=1)
        return weighted_output

    def forward(self, x, sen_len, doc_len, video_feat=None, audio_feat=None):
        batch_size = x.size(0)
        x = x.view(-1, self.max_sen_len, self.embedding_dim)
        sen_len_flat = sen_len.view(-1)

        word_features, _ = self.word_bilstm(x)
        word_features = self.dropout_layer(word_features)
        word_representation = self.attention(word_features, sen_len_flat)
        word_representation = word_representation.view(batch_size, self.max_utt, -1)

        if video_feat is not None:
            word_representation = torch.cat([word_representation, video_feat], dim=-1)
        if audio_feat is not None:
            word_representation = torch.cat([word_representation, audio_feat], dim=-1)

        sentence_features, _ = self.sentence_bilstm(word_representation)
        sentence_features = self.dropout_layer(sentence_features)

        emotion_representation = self.attention(sentence_features, doc_len)
        cause_representation = self.attention(sentence_features, doc_len)

        pred_emotion = self.fc_emotion(emotion_representation)
        pred_cause = self.fc_cause(cause_representation)

        return pred_emotion, pred_cause

# class TAVDataset(Dataset):
#     def __init__(self, data_file, tokenizer, word_idx, video_idx, max_doc_len, max_sen_len):
#         # Load data using helper function
#         x, x_v, sen_len, doc_len, y_emotion, y_cause, doc_id, y_pairs = load_data(
#             data_file, tokenizer, word_idx, video_idx, max_doc_len, max_sen_len
#         )
        
#         self.x = x
#         self.x_v = x_v
#         self.sen_len = sen_len
#         self.doc_len = doc_len
#         self.y_emotion = y_emotion
#         self.y_cause = y_cause
#         self.doc_id = doc_id
#         self.y_pairs = y_pairs

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         return (
#             torch.tensor(self.x[idx], dtype=torch.float),
#             torch.tensor(self.sen_len[idx], dtype=torch.long), 
#             torch.tensor(self.doc_len[idx], dtype=torch.long),
#             torch.tensor(self.y_emotion[idx], dtype=torch.long),
#             torch.tensor(self.y_cause[idx], dtype=torch.long)
#         )

class TAVDataset(Dataset):
    def __init__(self, data_file, tokenizer, word_idx, video_idx, max_doc_len, max_sen_len):
        # Load data using helper function
        x, x_v, sen_len, doc_len, y_emotion, y_cause, doc_id, y_pairs = load_data(
            data_file, tokenizer, word_idx, video_idx, max_doc_len, max_sen_len
        )
        
        self.x = x
        self.x_v = x_v
        self.sen_len = sen_len
        self.doc_len = doc_len
        self.y_emotion = y_emotion
        self.y_cause = y_cause
        self.doc_id = doc_id
        self.y_pairs = y_pairs

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float),
            torch.tensor(self.sen_len[idx], dtype=torch.long), 
            torch.tensor(self.doc_len[idx], dtype=torch.long),
            torch.tensor(self.y_emotion[idx], dtype=torch.long),
            torch.tensor(self.y_cause[idx], dtype=torch.long)
        )
    
def collate_fn(batch):
    x, sen_len, doc_len, y_emotion, y_cause = zip(*batch)
    x = torch.stack(x).to(args.device)
    sen_len = torch.stack(sen_len).to(args.device)
    doc_len = torch.stack(doc_len).to(args.device) 
    y_emotion = torch.stack(y_emotion).to(args.device)
    y_cause = torch.stack(y_cause).to(args.device)
    return x, sen_len, doc_len, y_emotion, y_cause


def train_model():
    args = Args()
    set_seed(42)

    # Load embeddings and mappings
    embedding, embedding_pos, word_idx, _ = load_w2v(
        args.word_emb_dim,
        args.pos_emb_dim, 
        './data/train.json',
        './data/glove.txt'
    )
    
    video_idx, video_embeddings, audio_embeddings = load_video_audio_embeddings(
        args.video_idx_file,
        args.video_emb_file,
        args.audio_emb_file
    )

    # Create datasets
    train_dataset = TAVDataset(
        './data/train.json',
        None, # No tokenizer for BiLSTM
        word_idx,
        video_idx,
        args.max_utt,
        args.max_words
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Initialize model
    model = BiLSTMModule(
        embedding_dim=args.word_emb_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.num_classes,
        max_sen_len=args.max_words,
        max_doc_len=args.max_utt,
        dropout=args.dropout
    ).to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lrate, weight_decay=args.l2_reg)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        y_true_emotion, y_pred_emotion = [], []
        y_true_cause, y_pred_cause = [], []

        for x, sen_len, doc_len, y_emotion, y_cause in train_loader:
            pred_emotion, pred_cause = model(x, sen_len, doc_len)
            
            loss = criterion(pred_emotion, y_emotion) + criterion(pred_cause, y_cause)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # epoch metrics
        p_e, r_e, f1_e = calculate_metrics(y_true_emotion, y_pred_emotion)
        p_c, r_c, f1_c = calculate_metrics(y_true_cause, y_pred_cause)
   
        #Collect predictions
        y_true_emotion.extend(y_emotion.cpu().numpy())
        y_pred_emotion.extend(torch.argmax(pred_emotion, dim=1).cpu().numpy())
        y_true_cause.extend(y_cause.cpu().numpy())
        y_pred_cause.extend(torch.argmax(pred_cause, dim=1).cpu().numpy())
     
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")
        print(f"Emotion - P: {p_e:.4f}, R: {r_e:.4f}, F1: {f1_e:.4f}")
        print(f"Cause - P: {p_c:.4f}, R: {r_c:.4f}, F1: {f1_c:.4f}")
        print("-" * 50)

    print("Training complete!")

if __name__ == "__main__":
    train_model()