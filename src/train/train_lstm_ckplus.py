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
INPUT_DIM = 768 # Feature dim do ConvNeXt Tiny
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES = 8 
BATCH_SIZE = 16
EPOCHS = 50

# 1. Preparação dos Dados Temporais
def prepare_sequences():
    features = np.load(FEATURES_PATH)
    labels = np.load(LABELS_PATH)
    with open(NAMES_PATH, "r") as f:
        filenames = [line.strip() for line in f.readlines()]

    sequences = []
    seq_labels = []
    
    current_seq = []
    last_id = ""

    for i, fname in enumerate(filenames):
        # Extrai ID da sequência (ex: S010_004)
        parts = os.path.basename(fname).split("_")
        seq_id = f"{parts[0]}_{parts[1]}"
        
        if seq_id != last_id and len(current_seq) > 0:
            sequences.append(np.array(current_seq))
            seq_labels.append(labels[i-1])
            current_seq = []
        
        current_seq.append(features[i])
        last_id = seq_id
    
    # Adiciona a última
    if len(current_seq) > 0:
        sequences.append(np.array(current_seq))
        seq_labels.append(labels[-1])

    return sequences, seq_labels

class FERSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Padding manual para simplificar (todas seqs do CK+ são curtas aqui)
        seq = self.sequences[idx]
        return torch.FloatTensor(seq), self.labels[idx]

# 2. Modelo LSTM
class FERLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(FERLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Pega apenas o output do último frame da sequência
        out = self.fc(out[:, -1, :])
        return out

def train():
    seqs, labels = prepare_sequences()
    X_train, X_test, y_train, y_test = train_test_split(seqs, labels, test_size=0.2, random_state=42)

    train_ds = FERSequenceDataset(X_train, y_train)
    test_ds = FERSequenceDataset(X_test, y_test)

    # Nota: Batch size 1 pq as sequências podem ter comprimentos diferentes nesta versão simplificada
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    model = FERLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"--- Treinando LSTM em {len(seqs)} sequências ---")
    
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long() # Cast para Long
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        # Eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).long() # Cast para Long
                out = model(x)
                _, pred = out.max(1)
                correct += (pred == y).item()
        
        acc = correct / len(test_ds)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {t_loss/len(train_ds):.4f} | Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/fer_lstm_ckplus.pth")

    print(f"Melhor Acurácia LSTM: {best_acc:.4f}")

if __name__ == "__main__":
    train()
