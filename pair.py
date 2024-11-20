import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from helper import load_data2, calculate_metrics 

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description='BiLSTM Training Configuration')

    # Embedding parameters
    parser.add_argument('--w2v_file', type=str, default='./data/ECF_glove_300.txt', help='Word embedding file')
    parser.add_argument('--path', type=str, default='./data/', help='Path for dataset')
    parser.add_argument('--video_emb_file', type=str, default='./data/video_embedding_4096.npy', help='Video embedding file')
    parser.add_argument('--audio_emb_file', type=str, default='./data/audio_embedding_6373.npy', help='Audio embedding file')
    parser.add_argument('--video_idx_file', type=str, default='./data/video_id_mapping.npy', help='Video index mapping file')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Dimension of word embedding')
    parser.add_argument('--embedding_dim_pos', type=int, default=50, help='Dimension of position embedding')

    # Input structure
    parser.add_argument('--max_sen_len', type=int, default=35, help='Max number of tokens per sentence')
    parser.add_argument('--pred_future_cause', type=int, default=1, help='Consider cause among future utterances')

    # Model structure
    parser.add_argument('--choose_emocate', type=str, default='', help='Whether to choose emocate')
    parser.add_argument('--emocate_eval', type=int, default=6, help='Emocate evaluation parameter')
    parser.add_argument('--use_x_v', type=str, default='', help='Use video embedding')
    parser.add_argument('--use_x_a', type=str, default='', help='Use audio embedding')
    parser.add_argument('--n_hidden', type=int, default=100, help='Number of hidden units')
    parser.add_argument('--n_class', type=int, default=2, help='Number of distinct classes')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=200, help='Number of examples per batch')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--keep_prob1', type=float, default=0.5, help='Dropout keep probability for word embedding')
    parser.add_argument('--keep_prob2', type=float, default=1.0, help='Dropout keep probability for softmax layer')
    parser.add_argument('--l2_reg', type=float, default=1e-5, help='L2 regularization lambda')
    parser.add_argument('--keep_prob_v', type=float, default=0.5, help='Dropout keep probability for visual features')
    parser.add_argument('--keep_prob_a', type=float, default=0.5, help='Dropout keep probability for audio features')
    parser.add_argument('--end_run', type=int, default=21, help='End run')
    parser.add_argument('--training_iter', type=int, default=12, help='Number of training iterations')
    parser.add_argument('--log_path', type=str, default='./log', help='Path to save logs')
    parser.add_argument('--scope', type=str, default='TEMP', help='Scope name')
    parser.add_argument('--log_file_name', type=str, default='step2.log', help='Log file name')
    parser.add_argument('--save_pair', type=str, default='yes', help='Whether to save predicted pairs')
    parser.add_argument('--step1_file_dir', type=str, default='step1/', help='Directory of step1 files')

    # Additional parameters
    parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding dimension')
    parser.add_argument('--pos_emb_dim', type=int, default=50, help='Position embedding dimension')
    parser.add_argument('--max_words', type=int, default=35, help='Maximum words per sentence')
    parser.add_argument('--max_utt', type=int, default=35, help='Maximum utterances per document')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')

    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

# Dataset Class
class EmotionCauseDataset(Dataset):
    def __init__(self, data_file_name, word_idx, video_idx, max_sen_len, pred_future_cause):
        x, sen_len, distance, x_v, y, pair_id_all, pair_id, doc_id_list, y_pairs = load_data2(
            data_file_name, word_idx, video_idx, max_sen_len, pred_future_cause
        )
        self.x = x
        self.sen_len = sen_len
        self.distance = distance
        self.x_v = x_v
        self.y = y
        self.pair_id_all = pair_id_all
        self.pair_id = pair_id
        self.doc_id_list = doc_id_list
        self.y_pairs = y_pairs

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.x[idx], dtype=torch.long),
            'sen_len': torch.tensor(self.sen_len[idx], dtype=torch.long),
            'distance': torch.tensor(self.distance[idx], dtype=torch.long),
            'x_v': torch.tensor(self.x_v[idx], dtype=torch.long),
            'y': torch.tensor(self.y[idx], dtype=torch.long)
        }

# Model Class
class BiLSTMModel(nn.Module):
    def __init__(self, word_embedding, pos_embedding, video_embedding, audio_embedding, hidden_dim, output_dim, dropout):
        super(BiLSTMModel, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_embedding, freeze=False)
        self.pos_embedding = nn.Embedding.from_pretrained(pos_embedding, freeze=False)
        self.video_embedding = nn.Embedding.from_pretrained(video_embedding, freeze=False)
        self.audio_embedding = nn.Embedding.from_pretrained(audio_embedding, freeze=False)
        self.lstm = nn.LSTM(input_size=word_embedding.size(1), hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Bidirectional LSTM
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sen_len, distance, x_v):
        inputs = self.word_embedding(x)
        inputs = self.dropout(inputs)
        x_v = self.video_embedding(x_v)
        x_v = self.dropout(x_v)
        lstm_out, _ = self.lstm(inputs)
        lstm_out = self.dropout(lstm_out)
        combined_features = torch.cat((lstm_out, x_v), dim=2)
        distance = self.pos_embedding(distance)
        combined_features = torch.cat((combined_features, distance), dim=2)
        output = self.fc(combined_features)
        return output

# Collate Function
def collate_fn(batch, device):
    x = torch.stack([item['x'] for item in batch]).to(device)
    sen_len = torch.stack([item['sen_len'] for item in batch]).to(device)
    distance = torch.stack([item['distance'] for item in batch]).to(device)
    x_v = torch.stack([item['x_v'] for item in batch]).to(device)
    y = torch.stack([item['y'] for item in batch]).to(device)
    return x, sen_len, distance, x_v, y

# Training Function
def train_model(args, train_loader, dev_loader, model, criterion, optimizer):
    best_dev_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        y_true, y_pred = [], []

        for batch in train_loader:
            x, sen_len, distance, x_v, y = batch
            optimizer.zero_grad()
            outputs = model(x, sen_len, distance, x_v)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Collect predictions
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        dev_loss = 0
        y_dev_true, y_dev_pred = [], []
        with torch.no_grad():
            for batch in dev_loader:
                x, sen_len, distance, x_v, y = batch
                outputs = model(x, sen_len, distance, x_v)
                loss = criterion(outputs, y)
                dev_loss += loss.item()
                y_dev_true.extend(y.cpu().numpy())
                y_dev_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        dev_loss /= len(dev_loader)

        # Calculate metrics
        p, r, f1 = calculate_metrics(y_dev_true, y_dev_pred)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Dev Loss: {dev_loss:.4f}")
        print(f"Pair - Precision: {p:.4f}, Recall: {r:.4f}, F1 Score: {f1:.4f}")
        print("-" * 50)

        # Save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), os.path.join(args.log_path, 'best_model.pth'))

    print("Training complete!")

# Main Function
def main():
    args = parse_args()
    set_seed(42)
    device = args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create log directory
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    # Load embeddings
    word_embedding = torch.tensor(np.load(args.w2v_file), dtype=torch.float32)
    pos_embedding_path = os.path.join(args.path, 'pos_embedding.npy')
    if not os.path.exists(pos_embedding_path):
        raise FileNotFoundError(f"Position embedding file not found at {pos_embedding_path}")
    pos_embedding = torch.tensor(np.load(pos_embedding_path), dtype=torch.float32)
    video_embedding = torch.tensor(np.load(args.video_emb_file), dtype=torch.float32)
    audio_embedding = torch.tensor(np.load(args.audio_emb_file), dtype=torch.float32)

    # Initialize Dataset and DataLoader
    train_dataset = EmotionCauseDataset(
        data_file_name=os.path.join(args.path, 'train.txt'),
        word_idx=None,  # Replace with actual word_idx if available
        video_idx=None,  # Replace with actual video_idx if available
        max_sen_len=args.max_sen_len,
        pred_future_cause=args.pred_future_cause
    )
    dev_dataset = EmotionCauseDataset(
        data_file_name=os.path.join(args.path, 'dev.txt'),
        word_idx=None,
        video_idx=None,
        max_sen_len=args.max_sen_len,
        pred_future_cause=args.pred_future_cause
    )
    test_dataset = EmotionCauseDataset(
        data_file_name=os.path.join(args.path, 'test.txt'),
        word_idx=None,
        video_idx=None,
        max_sen_len=args.max_sen_len,
        pred_future_cause=args.pred_future_cause
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, args.device)
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, args.device)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, args.device)
    )

    # Initialize model, criterion, optimizer
    model = BiLSTMModel(
        word_embedding=word_embedding,
        pos_embedding=pos_embedding,
        video_embedding=video_embedding,
        audio_embedding=audio_embedding,
        hidden_dim=args.hidden_dim,
        output_dim=args.num_classes,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)

    # Start training
    train_model(args, train_loader, dev_loader, model, criterion, optimizer)


if __name__ == "__main__":
    main()