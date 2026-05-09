import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Configurações
BASE_DIR = "."
FEATURES_PATH = os.path.join(BASE_DIR, "data/processed/features/ckplus_features.npy")
LABELS_PATH = os.path.join(BASE_DIR, "data/processed/features/ckplus_labels.npy")
NAMES_PATH = os.path.join(BASE_DIR, "data/processed/features/ckplus_filenames.txt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 768 
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES = 8 
BATCH_SIZE = 16
EPOCHS = 60

def prepare_sequences():
    features = np.load(FEATURES_PATH)
    labels = np.load(LABELS_PATH)
    with open(NAMES_PATH, "r") as f:
        filenames = [line.strip() for line in f.readlines()]
    sequences, seq_labels = [], []
    current_seq, last_id = [], ""
    for i, fname in enumerate(filenames):
        parts = os.path.basename(fname).split("_")
        seq_id = f"{parts[0]}_{parts[1]}"
        if seq_id != last_id and len(current_seq) > 0:
            sequences.append(np.array(current_seq))
            seq_labels.append(labels[i-1])
            current_seq = []
        current_seq.append(features[i])
        last_id = seq_id
    if len(current_seq) > 0:
        sequences.append(np.array(current_seq))
        seq_labels.append(labels[-1])
    return sequences, seq_labels

class FERSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), self.labels[idx]

# 2. Modelo LSTM com Attention
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_dim)
        attn_weights = torch.tanh(self.attn(lstm_output)) # (batch, seq_len, 1)
        soft_attn_weights = torch.softmax(attn_weights, dim=1)
        # Context vector: soma ponderada das hidden states
        context = torch.sum(soft_attn_weights * lstm_output, dim=1)
        return context, soft_attn_weights

class FERAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(FERAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = Attention(hidden_dim * 2) # *2 por ser bidirecional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        context, weights = self.attention(out)
        out = self.fc(context)
        return out

def train():
    seqs, labels = prepare_sequences()
    X_train, X_test, y_train, y_test = train_test_split(seqs, labels, test_size=0.2, random_state=42)
    train_ds = FERSequenceDataset(X_train, y_train)
    test_ds = FERSequenceDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = FERAttentionLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"--- Treinando LSTM + Attention ---")
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                out = model(x)
                _, pred = out.max(1)
                correct += (pred == y).item()
        
        acc = correct / len(test_ds)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {t_loss/len(train_ds):.4f} | Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/fer_attention_lstm_ckplus.pth")

    print(f"Melhor Acurácia LSTM+Attention: {best_acc:.4f}")

if __name__ == "__main__":
    train()
