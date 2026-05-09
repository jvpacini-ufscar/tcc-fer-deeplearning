import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configurações
BASE_DIR = "."
FEATURES_PATH = os.path.join(BASE_DIR, "data/processed/features/ckplus_features.npy")
LABELS_PATH = os.path.join(BASE_DIR, "data/processed/features/ckplus_labels.npy")
NAMES_PATH = os.path.join(BASE_DIR, "data/processed/features/ckplus_filenames.txt")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 768 
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES = 7 
BATCH_SIZE = 1 
EPOCHS = 40
EMOTION_CLASSES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

# 1. Preparação dos Dados por Sujeito
def prepare_subject_data():
    features = np.load(FEATURES_PATH)
    labels = np.load(LABELS_PATH)
    with open(NAMES_PATH, "r") as f:
        filenames = [line.strip() for line in f.readlines()]

    subjects_dict = {} # subject_id -> list of (sequence, label)
    
    current_seq = []
    last_seq_id = ""

    for i, fname in enumerate(filenames):
        parts = os.path.basename(fname).split("_")
        subject_id = parts[0]
        seq_id = f"{parts[0]}_{parts[1]}"
        
        if seq_id != last_seq_id and len(current_seq) > 0:
            if last_subject_id not in subjects_dict:
                subjects_dict[last_subject_id] = []
            subjects_dict[last_subject_id].append((np.array(current_seq), labels[i-1]))
            current_seq = []
        
        current_seq.append(features[i])
        last_seq_id = seq_id
        last_subject_id = subject_id
    
    # Última
    if len(current_seq) > 0:
        if last_subject_id not in subjects_dict:
            subjects_dict[last_subject_id] = []
        subjects_dict[last_subject_id].append((np.array(current_seq), labels[-1]))

    return subjects_dict

class FERSequenceDataset(Dataset):
    def __init__(self, data_list):
        self.sequences = [item[0] for item in data_list]
        self.labels = [item[1] for item in data_list]
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), self.labels[idx]

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
    def forward(self, lstm_output):
        attn_weights = torch.tanh(self.attn(lstm_output))
        soft_attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(soft_attn_weights * lstm_output, dim=1)
        return context

class FERAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(FERAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        context = self.attention(out)
        return self.fc(context)

def train_fold(fold_idx, train_data, val_data):
    model = FERAttentionLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    
    train_loader = DataLoader(FERSequenceDataset(train_data), batch_size=1, shuffle=True)
    val_loader = DataLoader(FERSequenceDataset(val_data), batch_size=1, shuffle=False)

    best_val_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        # Eval
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                out = model(x)
                _, pred = out.max(1)
                y_true.append(y.item())
                y_pred.append(pred.item())
        
        acc = accuracy_score(y_true, y_pred)
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), f"models/best_model_fold_{fold_idx}.pth")
            
    print(f"Fold {fold_idx} concluído. Melhor Acc: {best_val_acc:.4f}")
    return best_val_acc, y_true, y_pred

def main():
    subjects_dict = prepare_subject_data()
    subject_ids = list(subjects_dict.keys())
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    all_y_true, all_y_pred = [], []

    print(f"--- Iniciando 5-Fold Cross-Validation (Subject-Independent) ---")
    print(f"Total de Sujeitos: {len(subject_ids)}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(subject_ids)):
        train_subjects = [subject_ids[i] for i in train_idx]
        val_subjects = [subject_ids[i] for i in val_idx]
        
        train_data, val_data = [], []
        for s_id in train_subjects: train_data.extend(subjects_dict[s_id])
        for s_id in val_subjects: val_data.extend(subjects_dict[s_id])
        
        acc, y_t, y_p = train_fold(fold+1, train_data, val_data)
        fold_accuracies.append(acc)
        all_y_true.extend(y_t)
        all_y_pred.extend(y_p)

    print(f"\nAcurácia Média Final (LOSO-ish): {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    
    # Relatório Final
    report = classification_report(all_y_true, all_y_pred, target_names=EMOTION_CLASSES, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(REPORTS_DIR, "metrics_lstm_ckplus_rigorous.csv"))
    
    # Matriz de Confusão Acumulada
    cm = confusion_matrix(all_y_true, all_y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
    plt.title(f'Matriz de Confusão Rigorosa (Mean Acc: {np.mean(fold_accuracies):.4f})')
    plt.savefig(os.path.join(FIGURES_DIR, "cm_lstm_ckplus_rigorous.png"))
    plt.close()

if __name__ == "__main__":
    main()
