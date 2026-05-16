import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_PATH = os.path.join(BASE_DIR, "data/processed/features/ckplus_features.npy")
LABELS_PATH = os.path.join(BASE_DIR, "data/processed/features/ckplus_labels.npy")
NAMES_PATH = os.path.join(BASE_DIR, "data/processed/features/ckplus_filenames.txt")
MODEL_PATH = os.path.join(BASE_DIR, "models/fer_attention_lstm_ckplus.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports/figures/temporal_evolution")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 768 
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES = 8 
EMOTION_CLASSES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'neutral']

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

def load_sequences():
    features = np.load(FEATURES_PATH)
    labels = np.load(LABELS_PATH)
    with open(NAMES_PATH, "r") as f:
        filenames = [line.strip() for line in f.readlines()]

    sequences = []
    current_seq = []
    last_seq_id = ""
    last_label = -1

    for i, fname in enumerate(filenames):
        parts = os.path.basename(fname).split("_")
        seq_id = f"{parts[0]}_{parts[1]}"
        
        if seq_id != last_seq_id and len(current_seq) > 0:
            sequences.append({
                'id': last_seq_id,
                'features': np.array(current_seq),
                'label': last_label
            })
            current_seq = []
        
        current_seq.append(features[i])
        last_seq_id = seq_id
        last_label = labels[i]
    
    if len(current_seq) > 0:
        sequences.append({
            'id': last_seq_id,
            'features': np.array(current_seq),
            'label': last_label
        })

    return sequences

def analyze_temporal_evolution():
    # Carregar modelo
    model = FERAttentionLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    sequences = load_sequences()
    print(f"Total de sequências carregadas: {len(sequences)}")

    lengths = [len(seq['features']) for seq in sequences]
    print(f"Distribuição de tamanhos de sequência: Min={min(lengths)}, Max={max(lengths)}, Média={np.mean(lengths):.2f}")
    
    # Selecionar uma sequência representativa para cada emoção
    selected_sequences = {}
    found_labels = set([seq['label'] for seq in sequences])
    print(f"Labels encontrados no dataset: {found_labels}")
    
    for seq in sequences:
        label_idx = int(seq['label'])
        emotion = EMOTION_CLASSES[label_idx]
        if emotion not in selected_sequences:
            selected_sequences[emotion] = seq
        else:
            # Pegar a mais longa disponível para essa emoção
            if len(seq['features']) > len(selected_sequences[emotion]['features']):
                selected_sequences[emotion] = seq
    
    for emotion, seq in selected_sequences.items():
        print(f"Selecionada: {emotion} (seq {seq['id']}, len {len(seq['features'])})")


    for emotion, seq in selected_sequences.items():
        print(f"Analisando sequência {seq['id']} (Emoção: {emotion}, Frames: {len(seq['features'])})")
        
        probs_history = []
        
        # Simular o progresso temporal
        with torch.no_grad():
            for t in range(1, len(seq['features']) + 1):
                # Pegar prefixo até t
                input_seq = torch.FloatTensor(seq['features'][:t]).unsqueeze(0).to(DEVICE)
                logits = model(input_seq)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                probs_history.append(probs)
        
        probs_history = np.array(probs_history)
        
        # Plotar
        plt.figure(figsize=(12, 6))
        for i, class_name in enumerate(EMOTION_CLASSES):
            linewidth = 4 if i == seq['label'] else 1.5
            alpha = 1.0 if i == seq['label'] else 0.6
            plt.plot(range(1, len(probs_history) + 1), probs_history[:, i], 
                     label=class_name, linewidth=linewidth, alpha=alpha)
        
        plt.title(f"Evolução Temporal da Probabilidade - Sequência {seq['id']} (Verdade: {emotion})")
        plt.xlabel("Frame index (Início -> Ápice)")
        plt.ylabel("Probabilidade (Softmax)")
        plt.ylim(-0.05, 1.05)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f"evolution_{emotion}_{seq['id']}.png")
        plt.savefig(save_path)
        print(f"Gráfico salvo em: {save_path}")
        plt.close()

if __name__ == "__main__":
    analyze_temporal_evolution()
