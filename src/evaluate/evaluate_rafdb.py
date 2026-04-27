import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import timm

# Configurações
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data/raw/raf-db")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = os.path.join(BASE_DIR, "models/resnet50v2_pytorch_best.pth")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_CLASSES = 7
EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 1. Pipeline de Dados (Idêntico ao usado no treino do FER2013)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_resnet50v2():
    # Mesma arquitetura do resnet50v2_pytorch.py
    base_model = timm.create_model('resnetv2_50', pretrained=False, num_classes=0)
    model = nn.Sequential(
        base_model,
        nn.Flatten(),
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, NUM_CLASSES)
    )
    return model.to(DEVICE)

def evaluate_cross_dataset():
    print(f"[INFO] Iniciando avaliação Cross-Dataset (FER2013 -> RAF-DB)")
    print(f"[INFO] Usando dispositivo: {DEVICE}")

    # Carregar Dados
    if not os.path.exists(TEST_DIR):
        print(f"[ERRO] Diretório de teste não encontrado: {TEST_DIR}")
        return

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Carregar Modelo
    model = get_resnet50v2()
    if not os.path.exists(MODEL_PATH):
        print(f"[ERRO] Pesos do modelo não encontrados em: {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Inferência RAF-DB"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.cpu().tolist())

    # Métricas
    report = classification_report(y_true, y_pred, target_names=EMOTION_CLASSES, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    # Adicionar Balanced Accuracy
    b_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"\n[RESULTADO] Balanced Accuracy: {b_acc:.4f}")
    print(f"[RESULTADO] Accuracy (Global): {report['accuracy']:.4f}")

    # Salvar Relatório
    csv_path = os.path.join(REPORTS_DIR, "metrics_cross_fer2013_to_rafdb.csv")
    df_report.to_csv(csv_path)
    print(f"[INFO] Métricas salvas em: {csv_path}")

    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=EMOTION_CLASSES, yticklabels=EMOTION_CLASSES)
    plt.title('Matriz de Confusão Cross-Dataset: FER2013 -> RAF-DB')
    plt.ylabel('Verdadeiro (RAF-DB)')
    plt.xlabel('Predito (Modelo FER2013)')
    
    fig_path = os.path.join(FIGURES_DIR, "cm_cross_fer2013_to_rafdb.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Matriz de confusão salva em: {fig_path}")

if __name__ == "__main__":
    evaluate_cross_dataset()
