import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_PATH = os.path.join(BASE_DIR, "data/processed/features/rafdb_test_features.npy")
LABELS_PATH = os.path.join(BASE_DIR, "data/processed/features/rafdb_test_labels.npy")
MODEL_PATH = os.path.join(BASE_DIR, "models/fer_attention_lstm_ckplus.pth")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 768 
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES = 8 # O modelo salvo tem 8 saídas

# Mapeamento: RAF -> CK+ (Model Index)
# RAF Folder Index: 0:angry, 1:disgust, 2:fear, 3:happy, 4:neutral, 5:sad, 6:surprise
# CK+ Model Index: 0:anger, 1:contempt, 2:disgust, 3:fear, 4:happy, 5:sadness, 6:surprise, 7:neutral
RAF_TO_CK_MODEL = {
    0: 0, # angry -> anger
    1: 2, # disgust -> disgust
    2: 3, # fear -> fear
    3: 4, # happy -> happy
    4: 7, # neutral -> neutral (assumido)
    5: 5, # sad -> sadness
    6: 6  # surprise -> surprise
}
EMOTIONS_COMMON = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

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

def evaluate_cross_dataset():
    print("--- Iniciando Avaliação Cross-Dataset (CK+ Model -> RAF-DB Test) ---")
    
    # 1. Carregar Modelo
    model = FERAttentionLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Carregar Dados
    features = np.load(FEATURES_PATH)
    labels = np.load(LABELS_PATH)
    
    y_true, y_pred = [], []
    
    # 3. Inferência
    # RAF-DB é estático, então tratamos cada imagem como uma "sequência" de 1 frame
    with torch.no_grad():
        for i in range(len(features)):
            # Pular contempt se o RAF-DB não tiver (ele não tem)
            # Mapear label real do RAF para o índice que o modelo CK+ entende
            raf_label = labels[i]
            target_ck_label = RAF_TO_CK_MODEL[raf_label]
            
            x = torch.FloatTensor(features[i]).unsqueeze(0).unsqueeze(0).to(DEVICE) # (1, 1, 768)
            logits = model(x)
            _, pred = logits.max(1)
            
            y_true.append(target_ck_label)
            y_pred.append(pred.item())

    # 4. Métricas
    # Considerar apenas as classes que existem em ambos (removendo Contempt do report se necessário)
    # Mas vamos deixar o classification report lidar com isso
    target_names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'neutral']
    
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAcurácia Geral Cross-Dataset: {acc:.4f}")
    
    report = classification_report(y_true, y_pred, target_names=target_names, labels=range(8), output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(REPORTS_DIR, "metrics_cross_dataset_lstm_rafdb.csv"))
    
    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred, labels=range(8), normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Reds', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Matriz de Confusão Cross-Dataset (CK+ -> RAF-DB)\nAcc: {acc:.4f}')
    plt.savefig(os.path.join(FIGURES_DIR, "cm_cross_dataset_lstm_rafdb.png"))
    plt.close()
    
    print(f"Resultados salvos em {REPORTS_DIR}")

if __name__ == "__main__":
    evaluate_cross_dataset()
